"""
FMUKF Benchmarking Pipeline
===========================

This module orchestrates the complete benchmarking workflow for FMUKF and baseline estimators.
The pipeline follows this workflow:

1. **Worker Launch**: Spawn parallel workers (local processes or SLURM jobs) that:
   - Load test datasets (ship trajectories and environmental conditions)
   - Apply sensor configurations with noise
   - Run estimators (UKF variants, FMUKF, End2End models) on each test case
   - Save results to temporary H5 files (one per worker)

2. **Results Merging**: Combine all temporary H5 files into a single structured file with:
   - True/: Ground truth state trajectories
   - Sensors/: Sensor measurements with noise
   - Estimates/: Estimator predictions for comparison

3. **Error Analysis**: Compute Mean Absolute Error (MAE) for each estimator:
   - Compare predictions against ground truth
   - Handle angular wrapping for heading angle (psi)
   - Save results in a pandas dataframe as feather file

4. For debugging/convenience
    Runs Visualization script (with default parameters for convenience... you probably need to adjust its parameters during training)

Outputs:
   - benchmark.h5: Complete raw predictions for every estimator in every config
   - df_err.feather: Pandas dataframe with MAE error for every estimator

Configuration:
- Modify config/benchmark.yaml to adjust test scenarios, estimators, and compute resources
- Set launcher.type to "local" for development or "slurm" for HPC clusters
- Set device to "cpu", "cuda", or "auto" for model loading (see hardware recommendations)

Hardware Recommendations:
- For Ryzen 5600X + RTX 3070: Use "cpu" for small models, "cuda" for large models
- For HPC clusters: Use "auto" to automatically detect available devices

Usage:
    python benchmark.py  # Uses config/benchmark.yaml
    python benchmark.py launcher.local.num_workers=8  # Override config values
    python benchmark.py device=cuda  # Use GPU for model loading
    python benchmark.py no_progress=true  # Disable progress monitoring
"""


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

# Register all custom resolvers BEFORE importing hydra
from fmukf.utils.hydra import register_fmukf_resolvers
register_fmukf_resolvers()

import hydra
import subprocess
import sys
import re
import time
import json
import base64
from pathlib import Path
from omegaconf import DictConfig, OmegaConf


# Import benchmarking utilities
from fmukf.benchmarking.build_error_dataframe import merge_h5_files, compute_err_df_from_h5

@hydra.main(version_base=None, config_path="config", config_name="benchmark")
def main(cfg: DictConfig):
    """
    Main benchmarking orchestrator.
    
    Coordinates the complete benchmarking pipeline from worker launch through
    error analysis and output generation.
    
    Args:
        cfg: Hydra configuration containing all benchmark parameters
    """
    

    # ========================================
    # SETUP: Prepare directories and configuration
    # ========================================
    
    # Convert config to a plain dict and resolve all references
    config = OmegaConf.to_container(cfg, resolve=True)
    
    # Ensure all output directories exist
    os.makedirs(cfg.io.temp_stdout_folder, exist_ok=True)
    os.makedirs(cfg.io.temp_h5_folder, exist_ok=True)
    os.makedirs(os.path.dirname(cfg.io.merged_h5), exist_ok=True)
    os.makedirs(os.path.dirname(cfg.io.dest_feather), exist_ok=True)

    # Check for progress monitoring flag from config
    enable_progress = not cfg.get('no_progress', False)
    
    if not enable_progress:
        print("Progress monitoring disabled")
    print("="*60)
    print("FMUKF BENCHMARKING PIPELINE")
    print("="*60)
    print(f"Launcher: {cfg.launcher.type}")
    print(f"Output directory: {os.getcwd()}")
    print(f"Test environments: {cfg.data.num_max_envs}")
    print(f"Trajectories per env: {cfg.data.num_max_trajs}")
    print(f"Time steps per trajectory: {cfg.data.num_max_time_steps}")
    print(f"Sensor configurations: {len(cfg.sensors)}")
    print(f"Estimators: {len(cfg.estimators)}")
    print("-"*60)
    

    # Convert config to JSON then base64-encode it for worker processes
    cfg_json_str = json.dumps(config)
    cfg_b64_str = base64.b64encode(cfg_json_str.encode()).decode()

    # ========================================
    # EXECUTION: Launch workers and process results
    # ========================================
    
    launcher_type = cfg.launcher.type
    
    if launcher_type == "local":
        print("\n1. LAUNCHING LOCAL WORKERS...")
        launch_local_workers(cfg, cfg_b64_str, enable_progress)
        print("\n2. MERGING RESULTS...")
        merge_and_analyze_local(cfg)
    elif launcher_type == "slurm":
        print("\n1. LAUNCHING SLURM WORKERS...")
        job_id = launch_slurm_workers(cfg, cfg_b64_str)
        
        # # Check if we're using the devel partition which has a 2-job limit
        # is_devel_partition = cfg.launcher.slurm.partition.lower() == "devel"
        
        # if is_devel_partition:
        #     print("\n2. DEVEL PARTITION DETECTED - JOB LIMIT REACHED")
        #     print("   Will wait for worker jobs to complete and then run merge locally")
        #     print(f"   Monitor jobs with: squeue -j {job_id}")
        #     print("    After jobs complete, run: python wait_and_merge.py {job_id} {output_dir}")
        #     print(f"   Example: python wait_and_merge.py {job_id} {os.getcwd()}")
        # else:
        #     print("\n2. SUBMITTING MERGE/ANALYSIS JOB...")
        merge_and_analyze_slurm(cfg)
    else:
        raise ValueError(f"Unknown launcher type: {launcher_type}")
    
    print("\n" + "="*60)
    print("BENCHMARKING PIPELINE COMPLETE!")
    print("="*60)



# Global timing state for progress tracking
_progress_timing = {}

def read_progress_files(cfg: DictConfig, num_workers: int):
    """Read progress files from all workers and return progress data."""
    progress_data = {}
    progress_dir = Path(cfg.io.temp_stdout_folder)
    
    for rank in range(num_workers):
        progress_file = progress_dir / f"progress_rank_{rank}.json"
        if progress_file.exists():
            try:
                with open(progress_file, 'r') as f:
                    progress_data[rank] = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                progress_data[rank] = None
        else:
            progress_data[rank] = None
    
    return progress_data

def display_progress(cfg: DictConfig, num_workers: int):
    """Display progress bars for all workers with time estimates."""
    global _progress_timing
    progress_data = read_progress_files(cfg, num_workers)
    current_time = time.time()
    
    print("\n" + "="*60)
    print("WORKER PROGRESS")
    print("="*60)
    
    total_progress = 0
    active_workers = 0
    
    for rank in range(num_workers):
        data = progress_data.get(rank)
        if data is not None:
            percentage = data.get('percentage', 0)
            current = data.get('current', 0)
            total = data.get('total', 0)
            current_item = data.get('current_item', '')
            
            # Initialize timing state for this worker
            if rank not in _progress_timing:
                _progress_timing[rank] = {
                    'start_time': current_time,
                    'last_update': current_time,
                    'last_current': current,
                    'has_progress': False,
                    'last_eta': ""  # Store the last calculated ETA
                }
            
            timing = _progress_timing[rank]
            
            # Check if progress has increased (new work unit completed)
            progress_made = current - timing['last_current']
            if progress_made > 0:
                # Progress has increased - update timing and recalculate ETA
                timing['last_update'] = current_time
                timing['last_current'] = current
                timing['has_progress'] = True
                
                # Calculate ETA only when there's progress
                if current > 0:
                    total_elapsed = current_time - timing['start_time']
                    if total_elapsed > 0:
                        # Calculate average time per completed unit
                        avg_time_per_unit = total_elapsed / current
                        remaining_units = total - current
                        eta_seconds = avg_time_per_unit * remaining_units
                        
                        # Format ETA and STORE it
                        if eta_seconds < 60:
                            timing['last_eta'] = f"ETA: {eta_seconds:.0f}s"
                        elif eta_seconds < 3600:
                            eta_minutes = eta_seconds / 60
                            timing['last_eta'] = f"ETA: {eta_minutes:.0f}m"
                        else:
                            eta_hours = eta_seconds / 3600
                            timing['last_eta'] = f"ETA: {eta_hours:.1f}h"
            
            # Always use the stored ETA (this value is FIXED until next progress)
            time_estimate = timing.get('last_eta', "")
            
            # Create progress bar
            bar_length = 30
            filled_length = int(bar_length * percentage / 100)
            bar = '█' * filled_length + '░' * (bar_length - filled_length)
            
            # Add time estimate to output
            time_str = f" [{time_estimate}]" if time_estimate else ""
            print(f"Worker {rank:2d}: [{bar}] {percentage:5.1f}% ({current}/{total}) - {current_item}{time_str}")
            total_progress += percentage
            active_workers += 1
        else:
            print(f"Worker {rank:2d}: [{'░' * 30}]   0.0% (0/0) - Not started")
    
    if active_workers > 0:
        avg_progress = total_progress / active_workers
        print(f"\nAverage progress: {avg_progress:.1f}%")
    
    print("="*60)


def monitor_progress(cfg: DictConfig, num_workers: int, check_interval: float = 2.0):
    """Monitor progress of all workers and display updates."""
    global _progress_timing
    
    # Reset timing state when monitoring starts
    _progress_timing.clear()
    
    print(f"Monitoring progress of {num_workers} workers...")
    print("Press Ctrl+C to stop monitoring (workers will continue)")
    
    try:
        while True:
            display_progress(cfg, num_workers)
            time.sleep(check_interval)
    except KeyboardInterrupt:
        print("\nStopped monitoring. Workers will continue in background.")
        display_progress(cfg, num_workers)  # Final display



# ========================================
# LOCAL EXECUTION FUNCTIONS
# ========================================


def launch_local_workers(cfg: DictConfig, cfg_b64_str: str, enable_progress: bool = True):
    """
    Launch benchmark workers locally using subprocesses.
    
    For single worker: runs directly in current process (no subprocess overhead)
    For multiple workers: spawns parallel subprocesses
    
    Args:
        cfg: Hydra configuration
        cfg_b64_str: Base64-encoded configuration for worker processes
    """
    num_workers = cfg.launcher.local.num_workers
    
    print(f"Launching {num_workers} local workers...")
    
    if num_workers == 1:
        # Special case: run directly in current process (no subprocess overhead)
        print("Single worker - running directly in current process")
        
        # Import and run worker main function directly
        from fmukf.benchmarking.benchmark_worker import main_
        
        # Create a mock config with rank info
        config_dict = json.loads(base64.b64decode(cfg_b64_str).decode())
        config_dict["rank"] = 0
        config_dict["total_workers"] = 1
        worker_cfg = OmegaConf.create(config_dict)
        
        try:
            main_(worker_cfg, global_rank=0, global_tasks=1)
            print("Worker completed successfully!")
        except Exception as e:
            print(f"Worker failed: {e}")
            import traceback
            traceback.print_exc()
            raise
    else:
        # Multiple workers: use subprocesses
        processes = []
        
        for worker_id in range(num_workers):
            # Prepare stdout/stderr files
            stdout_file = Path(cfg.io.temp_stdout_folder) / f"worker_{worker_id}_stdout.txt"
            stderr_file = Path(cfg.io.temp_stdout_folder) / f"worker_{worker_id}_stderr.txt"
            
            # Launch subprocess with explicit rank and num_workers arguments
            cmd = [
                sys.executable, "-m", "fmukf.benchmarking.benchmark_worker", 
                cfg_b64_str,
                "--rank", str(worker_id),
                "--num-workers", str(num_workers)
            ]
            
            with open(stdout_file, 'w') as out, open(stderr_file, 'w') as err:
                process = subprocess.Popen(
                    cmd,
                    stdout=out,
                    stderr=err,
                    cwd=os.getcwd()
                )
            processes.append(process)
            print(f"  Started worker {worker_id} (PID: {process.pid})")
        
        # Monitor progress while waiting for workers to complete
        if enable_progress:
            # Reset timing state when monitoring starts
            global _progress_timing
            _progress_timing.clear()
            
            print("Monitoring worker progress...")
            print("Press Ctrl+C to stop monitoring (workers will continue)")
            
            try:
                while any(p.poll() is None for p in processes):
                    display_progress(cfg, num_workers)
                    time.sleep(2.0)
            except KeyboardInterrupt:
                print("\nStopped monitoring. Workers will continue in background.")
        else:
            print("Progress monitoring disabled - waiting for workers to complete...")
        
        # Wait for all workers to complete
        print("Waiting for all workers to complete...")
        failed_workers = []
        
        for i, process in enumerate(processes):
            return_code = process.wait()
            if return_code != 0:
                failed_workers.append(i)
                print(f"  Worker {i} failed with return code {return_code}")
            else:
                print(f"  Worker {i} completed successfully")
        
        if failed_workers:
            error_msg = f"ERROR: {len(failed_workers)} workers failed: {failed_workers}"
            print(error_msg)
            raise RuntimeError(error_msg)
        else:
            print("All workers completed successfully!")


def launch_slurm_workers(cfg: DictConfig, cfg_b64_str: str):
    """
    Launch benchmark workers using SLURM
    
    Returns:
        str: The job ID of the submitted job array
    """
    slurm_config = OmegaConf.to_container(cfg.launcher.slurm, resolve=True)
    io = OmegaConf.to_container(cfg["io"], resolve=True)
    
    print(f"Launching SLURM job array with {slurm_config['num_jobs']} jobs...")

    # Create SLURM batch script
    account_line = f"#SBATCH --account={slurm_config['account']}" if slurm_config.get('account') else ""
    sbatch_launcher_script = f"""#!/usr/bin/bash
#SBATCH --array=0-{int(slurm_config['num_jobs'])-1}
#SBATCH --time={slurm_config['time']}
#SBATCH --ntasks={slurm_config['num_tasks_per_job']}        
#SBATCH --cpus-per-task={slurm_config['cpus_per_task']}   
#SBATCH --nodes=1
#SBATCH --job-name=benchmark
{account_line}
#SBATCH --partition={slurm_config['partition']}
#SBATCH --output={io['temp_stdout_folder']}/sbatch_stdout_%A_%a.txt

# Activate environment (ie the conda or .venv activate script)
source {slurm_config.get('conda_activate_script', '/work/thes1788/miniforge3/envs/fmukf/activate_fmukf.sh')}

# Run worker
srun -n {slurm_config['num_tasks_per_job']} --kill-on-bad-exit=0 --output={io['temp_stdout_folder']}/srun_stdout_%A_%a_%t.txt python -m fmukf.benchmarking.benchmark_worker {cfg_b64_str}"""

    # Write batch script
    batch_script_path = Path(io['temp_stdout_folder']) / "launch_sbatch.sh"
    with open(batch_script_path, "w") as f:
        f.write(sbatch_launcher_script)

    # Submit job
    launch_cmd = f"sbatch {batch_script_path}"
    out = subprocess.check_output(launch_cmd, shell=True, text=True)
    print(out)
    
    # Extract and store job ID for merge step
    m = re.search(r"Submitted batch job (\d+)", out)
    if m:
        job_id = m.group(1)
        print(f"Primary job array submitted with job ID: {job_id}")
        
        # Store job_id in a file for merge step
        job_id_file = Path(io['temp_stdout_folder']) / "job_id.txt"
        with open(job_id_file, 'w') as f:
            f.write(job_id)
            
        return job_id
    else:
        raise RuntimeError(f"Failed to parse job ID from output: {out}")


def merge_and_analyze_local(cfg: DictConfig):
    """
    Merge worker results and compute error analysis locally.
    
    This function:
    1. Merges all temporary H5 files into a single structured file
    2. Computes MAE errors for all estimator predictions
    3. Saves error dataframe as feather file for analysis
    
    Args:
        cfg: Hydra configuration containing I/O paths
    """
    print("Merging H5 files...")
    merge_h5_files(cfg.io.temp_h5_folder, cfg.io.merged_h5)
    print(f"✓ Merged results saved to: {cfg.io.merged_h5}")
    
    print("Computing error analysis...")
    try:
        error_df = compute_err_df_from_h5(cfg.io.merged_h5)
        
        # Save as feather file for fast loading
        error_df.to_feather(cfg.io.dest_feather)
        print(f"✓ Error dataframe saved to: {cfg.io.dest_feather}")
        
        # Print summary statistics
        print(f"\nERROR ANALYSIS SUMMARY:")
        print(f"Total test cases: {len(error_df)}")
        print(f"Unique estimators: {error_df['estimator'].nunique()}")
        print(f"Unique environments: {error_df['env'].nunique()}")
        print(f"Unique sensor configs: {error_df['sensor'].nunique()}")
        
        # Show mean errors across all test cases
        state_cols = ['u', 'v', 'r', 'x', 'y', 'psi', 'p', 'phi', 'delta', 'n']
        mean_errors = error_df[state_cols].mean()
        print("\nMean errors across all test cases:")
        for state, error in mean_errors.items():
            print(f"  {state}: {error:.4f}")
            
    except Exception as e:
        print(f"ERROR in error analysis: {e}")
        import traceback
        traceback.print_exc()
        raise

    # Automatically visualize using the existing visualize.py and config
    try:
        run_visualization(cfg)
    except Exception as e:
        print(f"Visualization step failed: {e}")
        import traceback
        traceback.print_exc()


def merge_and_analyze_slurm(cfg: DictConfig):
    """
    Launch SLURM merge job that depends on worker jobs.
    
    Creates a dependent SLURM job that will run after all workers complete
    to merge results and perform error analysis.
    
    Args:
        cfg: Hydra configuration containing SLURM and I/O settings
    """
    slurm_config = OmegaConf.to_container(cfg.launcher.slurm, resolve=True)
    io = OmegaConf.to_container(cfg["io"], resolve=True)
    
    # Read job ID from file
    job_id_file = Path(io['temp_stdout_folder']) / "job_id.txt"
    if not job_id_file.exists():
        raise RuntimeError("Could not find job_id.txt - primary job may have failed")
    
    with open(job_id_file, 'r') as f:
        job_id = f.read().strip()
    
    print(f"Launching SLURM merge job depending on job {job_id}...")
    
    # Create merge job script
    account_line = f"#SBATCH --account={slurm_config['account']}" if slurm_config.get('account') else ""
    sbatch_merge_script = f"""#!/usr/bin/bash
#SBATCH --dependency=afterok:{job_id}
#SBATCH --time={slurm_config['time']}   
#SBATCH --cpus-per-task={slurm_config['cpus_per_task']}
#SBATCH --nodes=1
#SBATCH --job-name=fmukf_merge
{account_line}
#SBATCH --partition={slurm_config['partition']}
#SBATCH --output={io['temp_stdout_folder']}/merge_stdout.txt
#SBATCH --error={io['temp_stdout_folder']}/merge_stderr.txt

# Activate conda environment
source {slurm_config.get('conda_activate_script', '/work/thes1788/miniforge3/envs/fmukf/activate_fmukf.sh')}

# Run merge and error analysis
python -c "
import sys
from pathlib import Path
sys.path.append(str(Path.cwd()))

from fmukf.benchmarking.build_error_dataframe import merge_h5_files, compute_err_df_from_h5

print('Merging H5 files...')
merge_h5_files('{io['temp_h5_folder']}', '{io['merged_h5']}')
print('✓ H5 files merged')

print('Computing error analysis...')
error_df = compute_err_df_from_h5('{io['merged_h5']}')
error_df.to_feather('{io['dest_feather']}')
print('✓ Error analysis complete')

print(f'Results saved to:')
print(f'  H5 file: {io['merged_h5']}')
print(f'  Error dataframe: {io['dest_feather']}')
"

# Run visualization with overrides so outputs live under the benchmark run directory
python visualize.py \
  io.input_file='{io['dest_feather']}' \
  hydra_dir='{io['temp_stdout_folder']}/visualization'
"""

    # Write merge script
    merge_script_path = Path(io['temp_stdout_folder']) / "launch_merge_sbatch.sh"
    with open(merge_script_path, "w") as f:
        f.write(sbatch_merge_script)

    # Submit merge job
    launch_cmd = f"sbatch {merge_script_path}"
    out = subprocess.check_output(launch_cmd, shell=True, text=True)
    print(out)
    
    # Extract merge job ID
    m = re.search(r"Submitted batch job (\d+)", out)
    if m:
        merge_job_id = m.group(1)
        print(f"Merge job submitted with job ID: {merge_job_id}")
        print(f"Monitor with: squeue -u $USER")
        print(f"Check results in: {io['temp_stdout_folder']}/merge_stdout.txt")
    else:
        raise RuntimeError(f"Failed to parse merge job ID from output: {out}")


def merge_h5_files(source_dir: str, dest_file: str):
    """Simple H5 file merger"""
    try:
        import h5py
        
        source_path = Path(source_dir)
        dest_path = Path(dest_file)
        
        # Find all rank_*.h5 files
        rank_files = list(source_path.glob("rank_*.h5"))
        
        if not rank_files:
            print(f"No rank_*.h5 files found in {source_dir}")
            return
        
        print(f"Found {len(rank_files)} files to merge")
        
        # Create destination file
        with h5py.File(dest_path, 'w') as dest_f:
            # Initialize groups
            dest_f.create_group("True")
            dest_f.create_group("Sensors") 
            dest_f.create_group("Estimates")
            
            # Merge each rank file
            for rank_file in rank_files:
                print(f"  Merging {rank_file.name}...")
                
                with h5py.File(rank_file, 'r') as src_f:
                    # Copy all datasets from source to destination
                    def copy_datasets(name, obj):
                        if isinstance(obj, h5py.Dataset):
                            if name not in dest_f:
                                dest_f.create_dataset(name, data=obj[:])
                            else:
                                print(f"    Skipping duplicate dataset: {name}")
                    
                    src_f.visititems(copy_datasets)
        
        print(f"Merge complete: {dest_path}")
        
    except ImportError:
        print("h5py not available - skipping merge step")
    except Exception as e:
        print(f"Merge failed: {e}")


def run_visualization(cfg: DictConfig):
    """Run visualization script to generate plots and LaTeX table.

    This launches `Experiments/visualize.py` with overrides so that:
    - The input file is the just-created feather file from this benchmark run
    - The visualization hydra directory is a subfolder inside the benchmark run directory
    """
    print("\n3. VISUALIZING RESULTS...")
    print(f"  Using input: {cfg.io.dest_feather}")
    print(f"  Visualization output dir: {os.getcwd()}/visualization")

    # Build command with Hydra overrides
    cmd = [
        sys.executable,
        "visualize.py",
        f"io.input_file={cfg.io.dest_feather}",
        f"hydra_dir={os.getcwd()}/visualization",
    ]

    # Execute from the Experiments directory so visualize.py can find its config
    subprocess.run(cmd, check=True, cwd=str(Path(__file__).resolve().parent))
    print("✓ Visualization complete")


if __name__ == "__main__":
    main()
