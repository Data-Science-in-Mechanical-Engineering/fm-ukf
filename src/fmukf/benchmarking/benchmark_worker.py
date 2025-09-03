import os
# Brute force the max number of threads per process/worker 
NUM_THREADS = "1"
os.environ["OMP_NUM_THREADS"] = NUM_THREADS
os.environ["MKL_NUM_THREADS"] = NUM_THREADS
os.environ["NUMEXPR_NUM_THREADS"] = NUM_THREADS
os.environ["OPENBLAS_NUM_THREADS"] = NUM_THREADS
os.environ["VECLIB_MAXIMUM_THREADS"] = NUM_THREADS
os.environ["BLAS_NUM_THREADS"] = NUM_THREADS
os.environ["LAPACK_NUM_THREADS"] = NUM_THREADS
os.environ["MKL_DYNAMIC"] = "FALSE"
os.environ["OMP_DYNAMIC"] = "FALSE"

import sys
import json
import base64
import time
import h5py
import yaml
import hashlib
# import logging
import traceback
import numpy as np
from tqdm.auto import tqdm
from typing import Tuple, Dict
from pathlib import Path

from fmukf.benchmarking.configure_estimators import initEstimators, EstimatorWrapper

from omegaconf import DictConfig, OmegaConf
from fmukf.ML.models import MyTimesSeriesTransformer
from fmukf.ML.dataloader import H5LightningDataModule
from fmukf.simulation.container import ScaledContainer, ConstantVelocityContainer
from fmukf.simulation.envsimbase import EnvSimBase
from fmukf.simulation.sensors import SensorModelBase, MaskedFuzzyStateSensor

from func_timeout import func_timeout, FunctionTimedOut


#######################################
# PROGRESS TRACKING
#######################################
def update_progress(cfg: DictConfig, rank: int, current: int, total: int, current_item: str = ""):
    """Update progress file for this worker to track task completion.
    
    Creates or updates a JSON file in the temporary directory that contains information
    about the worker's progress. This file is used by the main process to monitor
    worker status and aggregate progress across all workers.
    
    Args:
        cfg: DictConfig
             Configuration object containing IO paths
        rank: int
             Worker rank/ID
        current: int
             Current number of completed tasks
        total: int
             Total number of tasks for this worker
        current_item: str
             Description of the current task being processed
    """
    progress_file = Path(cfg.io.temp_stdout_folder) / f"progress_rank_{rank}.json"
    progress_data = {
        "rank": rank,
        "current": current,
        "total": total,
        "percentage": (current / total * 100) if total > 0 else 0,
        "current_item": current_item,
        "timestamp": time.time()
    }
    
    with open(progress_file, 'w') as f:
        json.dump(progress_data, f)


def clear_progress(cfg: DictConfig, rank: int):
    """Clear progress file for this worker.
    
    Deletes the progress file for this worker if it exists. This is typically
    called at the beginning of worker execution to ensure a clean slate.
    
    Args:
        cfg: DictConfig
             Configuration object containing IO paths
        rank: int
             Worker rank/ID to identify the progress file
    """
    progress_file = Path(cfg.io.temp_stdout_folder) / f"progress_rank_{rank}.json"
    if progress_file.exists():
        progress_file.unlink()


#######################################
# HASH-BASED SEEDING
#######################################
def generate_instance_seed(*args) -> int:
    """Generate a unique seed based on a hash of input arguments. This is used to
    initialize the random number generator for each estimator.
    
    Creates a deterministic but unique seed for random number generators based on
    the provided arguments. This ensures reproducible randomness for each unique
    combination of inputs (e.g., environment, trajectory, sensor, and repetition).
    
    Args:
        *args: Any
             Variable arguments to hash together for seed generation
             
    Returns:
        int: A 32-bit integer seed derived from the hash of the input arguments
    """
    hash_object = hashlib.sha256(str(args).encode())
    hash_int = int.from_bytes(hash_object.digest()[:8], byteorder='big')
    return hash_int % (2**32)


#######################################
# SENSORS
#######################################
def initSensors(cfg: DictConfig) -> Dict[str, 'SensorModelBase']:
    """Initialize sensor models based on configuration.
    
    Creates sensor model instances based on the configuration. Currently supports
    MaskedFuzzyStateSensor type sensors which add noise to observed states.
    
    Args:
        cfg: DictConfig
             Configuration object containing sensor specifications and noise parameters
             
    Returns:
        Dict[str, SensorModelBase]: Dictionary mapping sensor keys to initialized sensor objects
        
    Raises:
        ValueError: If an invalid sensor type is specified in the configuration
    """
    sensors = {}
    for sensor_cfg in cfg.sensors:
        if sensor_cfg.type == "MaskedFuzzyStateSensor":
            global_stds = cfg.sensor_stds
            observed_states = sensor_cfg.observed_states
            stds = {state: global_stds[state] for state in observed_states}

            sensors[sensor_cfg.key] = MaskedFuzzyStateSensor(
                EnvSim = ScaledContainer(), # MaskedFuzzyState Sensor is independent of the environment/ship used (in this benchmark)
                std_dict = stds
            )
        else:
            raise ValueError(f"Invalid sensor type {sensor_cfg.type}")
    return sensors


#######################################
# H5 LOADING & SAVING
#######################################
def loadEnvXU(cfg: DictConfig, env_key: str, traj_key: str) -> Tuple['ScaledContainer', np.ndarray, np.ndarray]:
    """Load environment model and state/control trajectories from H5 file.
    
    Opens the H5 dataset and loads the true state trajectory X and control input
    trajectory U for the specified environment and trajectory keys. Also initializes
    the environment model (ship model) with parameters from the H5 file.
    
    Args:
        cfg: DictConfig
             Configuration object containing dataset path and parameters
        env_key: str
             Environment key in the H5 file
        traj_key: str
             Trajectory key within the environment group
             
    Returns:
        Tuple containing:
            ScaledContainer: Initialized environment model
            np.ndarray: State trajectory X with shape (L, 10) where L is the number of timesteps
            np.ndarray: Control input trajectory U with shape (L, 2)
    """
    with h5py.File(cfg.data.data_set.h5_path, "r") as f:
        X = np.array(f[env_key][traj_key]["x"][-cfg.data.num_max_time_steps:])
        U = np.array(f[env_key][traj_key]["u"][-cfg.data.num_max_time_steps:])
        parameters = yaml.safe_load(f[env_key].attrs["parameters"])
        env = ScaledContainer(parameters)
    return env, X, U


def saveEstimate(
    cfg: DictConfig,
    env_key: str,
    traj_key: str,
    sensor_key: str,
    repeat_key: int,
    estimator_key: str,
    xhat: np.ndarray,
    X: np.ndarray,
    U: np.ndarray,
    Y: np.ndarray,
):
    """Save estimation results and ground truth data to temporary H5 file.
    
    Writes the estimator's output (xhat), along with true state (X), control input (U),
    and sensor measurements (Y) to a temporary H5 file specific to this worker.
    The data is organized in a hierarchical structure for later merging and analysis.
    
    Args:
        cfg: DictConfig
             Configuration object containing IO paths
        env_key: str
             Environment key identifier
        traj_key: str
             Trajectory key identifier
        sensor_key: str
             Sensor key identifier
        repeat_key: int
             Repetition number for the sensor
        estimator_key: str
             Estimator key identifier
        xhat: np.ndarray
             Estimated state trajectory with shape (L, 10)
        X: np.ndarray
             True state trajectory with shape (L, 10)
        U: np.ndarray
             Control input trajectory with shape (L, 2)
        Y: np.ndarray
             Sensor measurement trajectory with shape (L, d_sensor)
    """

    h5_temp_path = os.path.join(cfg.io.temp_h5_folder, f"rank_{cfg.rank}.h5")
    with h5py.File(h5_temp_path, "a") as f:
        true_group = f.require_group("True")
        true_key = f"{env_key}/{traj_key}"
        if true_key not in true_group:
            true_group.create_dataset(true_key + "/X", data=X)
            true_group.create_dataset(true_key + "/U", data=U)

        sensors_group = f.require_group("Sensors")
        Y_key = "/".join(map(str, [env_key, traj_key, sensor_key, repeat_key]))
        if Y_key not in sensors_group:
            sensors_group.create_dataset(Y_key, data=Y)

        estimates_group = f.require_group("Estimates")
        Xhat_key = "/".join(map(str, [env_key, traj_key, sensor_key, repeat_key, estimator_key]))
        if Xhat_key not in estimates_group:
            estimates_group.create_dataset(Xhat_key, data=xhat)


#######################################
# WORKLOAD DISTRIBUTION
#######################################

def distributeEnvTrajKeys(cfg: DictConfig, rank: int, total_workers: int) -> list[tuple[str, str]]:
    """Distribute the workload of environment and trajectory pairs to each worker.
    
    Loads the test environment keys from the dataset and creates pairs of environment
    and trajectory keys. These pairs are then distributed evenly across workers using
    round-robin assignment based on worker rank.
    
    Args:
        cfg: DictConfig
             Configuration object containing dataset parameters
        rank: int
             Worker rank used to determine which subset of pairs to process
        total_workers: int
             Total number of workers for workload distribution
             
    Returns:
        list[tuple[str, str]]: List of (env_key, traj_key) pairs assigned to this worker
    """

    # Get the test keys from the Datamodule (!)
    h5dataset = H5LightningDataModule(h5_file_path_str=cfg.data.data_set.h5_path,
                                      num_envs_test=cfg.data.data_set.num_envs_test,
                                      num_envs_val= cfg.data.data_set.num_envs_val,
                                      seed=cfg.data.data_set.seed,
                                      compute_statistics = False,
                                      batch_size = 1,
                                      )
    _, _, test_env_keys = h5dataset.setup("getKeys")

    # Limit number of envs if specified
    if cfg.data.num_max_envs is not None:
        env_keys = test_env_keys[: cfg.data.num_max_envs]
    else:
        env_keys = test_env_keys

    # Limit number of trajectories per env if specified
    with h5py.File(cfg.data.data_set.h5_path, "r") as f:
        num_trajs_h5 = len(f[env_keys[0]])
        if cfg.data.num_max_trajs is None:
            num_trajs = num_trajs_h5
        else:
            num_trajs = min(cfg.data.num_max_trajs, num_trajs_h5)

    # Combine and distribute
    env_traj_pairs = []
    for ek in env_keys:
        for t in range(num_trajs):
            env_traj_pairs.append((ek, f"traj_{t}"))

    return env_traj_pairs[rank::total_workers]

#######################################
# ERROR HANDLING
#######################################
def log_error(cfg: DictConfig, error_type: str, context: Dict[str, str], exception_info: str = ""):
    """Log errors consistently to an error file.
    
    Creates or appends to an error log file for the worker, recording information
    about errors encountered during execution. This includes the error type,
    contextual information about where the error occurred, and optional exception
    details.
    
    Args:
        cfg: DictConfig
             Configuration object containing IO paths
        error_type: str
             Type/category of error (e.g., "Timeout", "Known Exception")
        context: Dict[str, str]
             Dictionary of contextual information about the error (e.g., environment, trajectory)
        exception_info: str
             Optional exception details or traceback information
    """
    error_file = Path(cfg.io.temp_stdout_folder) / f"AHHH_rank_{cfg.rank}.txt"
    
    # Build context string
    context_str = ", ".join([f"{k}={v}" for k, v in context.items()])
    
    with open(error_file, "a") as f:
        f.write(f"{error_type} in {context_str}\n")
        if exception_info:
            f.write(exception_info + "\n")


#######################################
# MAIN WORKER LOGIC
#######################################
def main():
    """Main entry point for the benchmark worker process.
    
    Parses command line arguments, determines execution context (SLURM vs local),
    sets up configuration, and delegates to main_() for actual benchmarking work.
    Also handles top-level exception catching and reporting.
    
    Command line arguments:
        config_b64: Base64-encoded configuration JSON string
        --rank: Worker rank for local execution
        --num-workers: Total number of workers for local execution
    """
    # -------------------------------------------------
    # PARSE ARGUMENTS
    # -------------------------------------------------
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark worker")
    parser.add_argument("config_b64", help="Base64-encoded configuration")
    parser.add_argument("--rank", type=int, help="Worker rank (for local execution)")
    parser.add_argument("--num-workers", type=int, help="Total number of workers (for local execution)")
    
    args = parser.parse_args()
    
    # -------------------------------------------------
    # DETERMINE EXECUTION CONTEXT (SLURM vs LOCAL)
    # -------------------------------------------------
    if args.rank is not None and args.num_workers is not None:
        # LOCAL EXECUTION
        global_rank = args.rank
        global_tasks = args.num_workers
        print(f"Running in LOCAL mode: rank {global_rank}/{global_tasks}")
    else:
        # SLURM EXECUTION - infer from environment
        array_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
        local_rank = int(os.environ.get("SLURM_PROCID", 0))
        local_tasks = int(os.environ.get("SLURM_NTASKS", 1))
        global_rank = array_id * local_tasks + local_rank
        global_tasks = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1)) * local_tasks
        print(f"Running in SLURM mode: rank {global_rank}/{global_tasks}")

    # -------------------------------------------------
    # READ AND PREPARE CONFIG
    # -------------------------------------------------
    cfg_b64_str = args.config_b64
    cfg_json_str = base64.b64decode(cfg_b64_str).decode()
    cfg_dict = json.loads(cfg_json_str)
    cfg_dict["rank"] = global_rank
    cfg_dict["total_workers"] = global_tasks
    cfg = OmegaConf.create(cfg_dict)

    # -------------------------------------------------
    # RUN MAIN WORKER LOGIC
    # -------------------------------------------------
    try:
        main_(cfg, global_rank, global_tasks)
    except:
        print(f"FATAL ERROR in worker {global_rank}!")
        print(f"Traceback:", traceback.format_exc())
        with open(cfg.io.temp_stdout_folder + f"/ERROR_rank_{cfg.rank}.txt", "a") as f:
             f.write("FATAL ERROR:" + traceback.format_exc())
        sys.exit(1)
    else:
        print(f"Worker {global_rank} completed successfully :)")
    finally:
        print(f"Worker {global_rank} reached end of main")


def main_(cfg: DictConfig, global_rank: int, global_tasks: int):
    """Main worker logic for running benchmarks.
    
    Distributes workload, initializes sensors and estimators, and processes assigned
    environment/trajectory pairs. For each pair, it generates measurements using sensors,
    applies estimators, and saves results to temporary files. Includes robust error
    handling with retries for estimator failures.
    
    Args:
        cfg: DictConfig
             Configuration object containing benchmark parameters
        global_rank: int
             Worker rank/ID among all workers
        global_tasks: int
             Total number of workers across all nodes
    """

    # Distribute Work (env/traj pairs)
    env_traj_keys = distributeEnvTrajKeys(cfg, global_rank, global_tasks)
    print(f"env_traj_keys for rank {global_rank}: {env_traj_keys}")
    if not env_traj_keys:
        print(f"rank {global_rank}: No env/trajectory pairs to process. Exiting.")
        return
    
    # Initialize Sensor and Estimators
    sensors = initSensors(cfg)
    estimators: Dict[str, EstimatorWrapper] = initEstimators(cfg)

    # Bookkeeping 
    total_items = len(env_traj_keys)
    clear_progress(cfg, global_rank)  # This file tells the master that this worker is alive
    

    for idx, (env_key, traj_key) in enumerate(env_traj_keys):  # Loop over env/traj pairs
        
        # Bookkeeping: Update progress file
        current_item = f"{env_key}/{traj_key}"
        update_progress(cfg, global_rank, idx + 1, total_items, current_item)
        

        # Load environment (ship model) and true state  X (shape L,10) and control input U (shape L,2) sequences
        env, X, U = loadEnvXU(cfg, env_key, traj_key)
        
        # Loop over sensors and repeats
        for sensor_key, sensor in sensors.items():
            for repeat_key in range(cfg.data.num_sensor_repeats):
                
                # Apply Sensor Model and get Measurements Y (shape L,d_sensor)
                seed = generate_instance_seed(cfg.base_seed, env_key, traj_key, sensor_key, repeat_key)
                sensor.reset_rng(seed)
                Y = np.array([sensor(x, u) for x, u in zip(X, U)])
                assert not np.isnan(Y).any(), f"Sensor returned NaN values (env={env_key}, traj={traj_key}, sensor={sensor_key}, repeat={repeat_key}). Skipping."
                assert isinstance(Y, np.ndarray), f"Sensor returned non-ndarray type {type(Y)} (env={env_key}, traj={traj_key}, sensor={sensor_key}, repeat={repeat_key}). Skipping."

                # Wrap Heading angle (psi) to be within [0,360) for X and Y
                psi_idx = env.state_vec_order.index("psi")
                X[:, psi_idx] = X[:, psi_idx] % 360
                if "psi" in sensor.y_vec_order:
                    psi_idx = sensor.y_vec_order.index("psi")
                    Y[:, psi_idx] = Y[:, psi_idx] % 360
            
                # Loop over all Estimator Models
                for estimator_key, estimator in estimators.items():
                    print(f"Estimating env={env_key}, traj={traj_key}, sensor={sensor_key}, repeat={repeat_key}, estimator={estimator_key}")
                    
    
                    # Set context for retry and error handling
                    success = False
                    error_context = {
                        "env_key": env_key, 
                        "traj_key": traj_key, 
                        "sensor_key": sensor_key, 
                        "repeat_key": repeat_key, 
                        "estimator_key": estimator_key
                    }
                    num_retries = cfg.error_handling.estimators.num_retries
                    retry_delay = cfg.error_handling.estimators.delay
                    timeout_seconds = cfg.error_handling.estimators.timeout

                    for attempt in range(num_retries + 1):  # +1 for initial attempt
                        if attempt > 0:
                            print(f"Retry attempt {attempt}/{num_retries} for {estimator_key}")
                            if retry_delay > 0:
                                time.sleep(retry_delay)
                        
                        try:
                            # Apply Estimator Model to get Estimate Xhat (shape L,10) with Timeout
                            Xhat = func_timeout(timeout_seconds, estimator.estimate, args=(env, sensor, Y, U, X, cfg.data.data_set.h))
                            
                            # Validate the shape, contents, etc.
                            assert isinstance(Xhat, np.ndarray), f"Estimator {estimator_key} returned non-ndarray type {type(Xhat)}"
                            assert Xhat.shape == X.shape, f"Estimator {estimator_key} returned shape {Xhat.shape} but expected {X.shape}"
                            assert not np.isnan(Xhat).any(), f"Estimator {estimator_key} returned NaN values"
                            success = True
                            break  # Success, exit retry loop
                            
                        except FunctionTimedOut: # Estimator timed out (likely RK45 Solver hung)
                            log_error(cfg, "Estimator timeout", error_context, f"Attempt {attempt + 1}/{num_retries + 1}")
                            if attempt == num_retries:  # Last attempt failed
                                break
                            
                        except Exception as e:   # Known Exception
                            log_error(cfg, "Known Exception", error_context, f"Attempt {attempt + 1}/{num_retries + 1}\n{traceback.format_exc()}")
                            if attempt == num_retries:  # Last attempt failed
                                break
                            
                        except:  # Unknown Exception (yes these apparrently exist sometimes)
                            log_error(cfg, "Unknown Error", error_context, f"Attempt {attempt + 1}/{num_retries + 1}\n{traceback.format_exc()}")
                            if attempt == num_retries:  # Last attempt failed
                                break

                    # If everything succeeded, do whatever you'd normally do (e.g., save results)
                    if success:
                        print("Success!")
                        saveEstimate(cfg, env_key, traj_key, sensor_key, repeat_key, estimator_key, Xhat, X, U, Y)
                    else:
                        print(f"Failed to estimate env={env_key}, traj={traj_key}, sensor={sensor_key}, repeat={repeat_key}, estimator={estimator_key}")
                        log_error(cfg, "Failed to estimate", error_context, f"Attempt {attempt + 1}/{num_retries + 1}\n{traceback.format_exc()}")
                        
    # Bookkeeping: Update progress file
    update_progress(cfg, global_rank, total_items, total_items, "Done")


    print(f"Rank {global_rank} is done with all env/traj pairs.")


if __name__ == "__main__":
    main() 