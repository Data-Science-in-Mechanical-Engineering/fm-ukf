import h5py
import pandas as pd
import numpy as np
import fmukf
from pathlib import Path


def merge_h5_files(source_dir: str, dest_file: str) -> None:
    """
    Merges every worker's temporary h5 file into a single file with hierarchical structure.
    
    The merged h5 file contains three main groups organized hierarchically:
    
    1. True/ - Ground truth data
       └── env_key/
           └── traj_key/
               ├── X (Dataset): True state sequences, shape (time_steps, 10)
               │   State vector: [u, v, r, x, y, psi, p, phi, delta, n]
               └── U (Dataset): Control inputs, shape (time_steps, 2)
    
    2. Sensors/ - Sensor measurement data
       └── env_key/
           └── traj_key/
               └── sensor_key/ (e.g., "ALL", "GuessUV")
                   └── repeat_num/ (e.g., "0", "1", "2")
                       └── (Dataset): Sensor measurements, shape (time_steps, 7)
    
    3. Estimates/ - Estimator predictions
       └── env_key/
           └── traj_key/
               └── sensor_key/ (e.g., "ALL", "GuessUV")
                   └── repeat_num/ (e.g., "0", "1", "2")
                       └── estimator_key/ (e.g., "BaseUKF__Q0.1__P1.0__False")
                           └── (Dataset): Estimated state sequences, shape (time_steps, 10)

    Args:
        source_dir: Path to the directory containing the temporary h5 files (rank_*.h5)
        dest_file: Path to the destination merged file

    Raises:
        RuntimeError: If no source files found or merge fails
    """
    try:
        source_path = Path(source_dir)
        dest_path = Path(dest_file)
        
        # Find all rank_*.h5 files
        rank_files = list(source_path.glob("rank_*.h5"))
        
        if not rank_files:
            raise RuntimeError(f"No rank_*.h5 files found in {source_dir}")
        
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
            
    except Exception as e:
        raise RuntimeError(f"Merge failed: {e}") from e


def get_leaf_keys(group: h5py.Group) -> list[str]:
    """
    Depth First Search all leaf datasets under a group and returns their keys.
    
    Args:
        group: HDF5 group to search
        
    Returns:
        List of dataset paths relative to the group
    """
    keys = []
    def visitobj(name, obj):
        if isinstance(obj, h5py.Dataset):
            keys.append(name)
    group.visititems(visitobj)
    return keys


def compute_err_per_dim(X: np.ndarray, Xhat: np.ndarray) -> np.ndarray:
    """
    Computes Mean Absolute Error (MAE) for each state dimension with angular wrapping.
    
    Special handling for heading angle (psi) to account for angular wrapping:
    - Errors are wrapped to [-180°, 180°] to always take the shortest angular path
    
    Args:
        X: True state sequence, shape (L, 10)
        Xhat: Estimated state sequence, shape (L, 10)
        
    Returns:
        MAE error for each dimension, shape (10,)
        
    Note:
        State vector order: [u, v, r, x, y, psi, p, phi, delta, n]
        where psi (index 5) is the heading angle requiring wrap handling.
    """
    err = X - Xhat

    # Wrap the heading angle error to [-180, 180] degrees (shortest angular path)
    angle_dim = 5  # psi index in fmukf.simulation.container.ScaledContainer().state_vec_order
    err[:, angle_dim] = (err[:, angle_dim] + 180) % 360 - 180

    # Compute MAE across time dimension
    err = np.abs(err)
    return err.mean(axis=0)

def compute_err_df_from_h5(h5_path: str) -> pd.DataFrame:
    """
    Computes MAE errors for all estimator predictions and returns as a structured dataframe.
    
    This function processes benchmark results stored in an h5 file and computes Mean Absolute 
    Error (MAE) for each estimator's state predictions compared to ground truth. Special 
    handling is applied to the heading angle (psi) to account for angular wrapping.
    
    Args:
        h5_path: Path to the benchmark h5 file containing estimates and ground truth data
        
    Returns:
        pd.DataFrame with the following structure:
        
        Metadata columns (str):
        - 'env': Environment identifier (e.g., "env_163", "env_28")
        - 'traj': Trajectory identifier (e.g., "traj_0", "traj_1") 
        - 'sensor': Sensor configuration (e.g., "ALL", "GuessUV")
        - 'repeat': Repeat number for noise initialization (e.g., "0", "1", "2")  
        - 'estimator': Estimator identifier (e.g., "BaseUKF__Q0.1__P1.0__False")
        - 'h5_key': Full h5 path to the estimate dataset
        
        Error columns (float64) - MAE for each state dimension:
        - 'u': Surge velocity error [m/s]
        - 'v': Sway velocity error [m/s] 
        - 'r': Yaw rate error [deg] (scaled from rad/s)
        - 'x': North position error [km] (scaled from m)
        - 'y': East position error [km] (scaled from m)
        - 'psi': Yaw angle error [deg] (angle-wrapped, scaled from rad)
        - 'p': Roll rate error [deg] (scaled from rad/s)
        - 'phi': Roll angle error [deg] (scaled from rad)
        - 'delta': Rudder angle error [deg] (scaled from rad)
        - 'n': Shaft velocity error [rpm] (scaled from unitless)
        
        Each row represents one estimator's performance on a specific test configuration
        (environment × trajectory × sensor setup × repeat). Error values are MAE over
        the entire trajectory length.
        
    Raises:
        RuntimeError: If h5 file cannot be opened or data structure is invalid
        KeyError: If expected datasets are missing from h5 file
    """
    try:
        with h5py.File(h5_path, 'r') as f:
            # Get all the keys for the estimates' predictions
            Xhat_keys = get_leaf_keys(f['Estimates'])  # ["env_key/traj_key/sensor_key/repeat_num/estimator_key", ...]    

            # Get all the keys for the ground truth states
            X_keys = get_leaf_keys(f['True'])  # ["env_key/traj_key/X", ...]
            X_keys = [key.split('/X')[0] for key in X_keys if key.endswith('/X')]  # Remove '/X' suffix

            # Sanity check: ensure ground truth exists for every estimate
            missing_truth = []
            for Xhat_key in Xhat_keys:
                try:
                    env_key, traj_key, sensor_key, repeat_key, estimator_key = Xhat_key.split('/')
                    env_traj_key = f'{env_key}/{traj_key}'
                    if env_traj_key not in X_keys:
                        missing_truth.append(env_traj_key)
                except ValueError as e:
                    raise ValueError(f"Invalid estimate key format: {Xhat_key}") from e
            
            if missing_truth:
                raise KeyError(f"Missing ground truth for: {missing_truth}")

            # Create dataframe with metadata columns
            key_tuples = [key.split('/') + [key] for key in Xhat_keys]
            err_df = pd.DataFrame(key_tuples, columns=['env', 'traj', 'sensor', 'repeat', 'estimator', 'h5_key'])

            # Get state vector ordering
            vec_order = fmukf.simulation.container.ScaledContainer().state_vec_order  # ['u', 'v', 'r', 'x', 'y', 'psi', 'p', 'phi', 'delta', 'n']

            # Efficiently compute errors for all dimensions and estimators
            def compute_all_errors(row):
                """Compute errors for all dimensions at once for a single row"""
                try:
                    # Build paths to estimate and ground truth
                    est_path = "/".join(["Estimates", row['env'], row['traj'], row['sensor'], row['repeat'], row['estimator']])
                    truth_path = "/".join(["True", row['env'], row['traj'], "X"])
                    
                    # Load data
                    Xhat = f[est_path][()]
                    X = f[truth_path][()]
                    
                    return compute_err_per_dim(X, Xhat)
                except KeyError as e:
                    print(f"Warning: Missing data for {row['h5_key']}: {e}")
                    return np.full(len(vec_order), np.nan)
            
            print("Computing errors for all dimensions and estimators...")
            all_errors = err_df.apply(compute_all_errors, axis=1, result_type='expand')
            
            # Assign each dimension's errors to corresponding columns
            for i, feature in enumerate(vec_order):
                err_df[feature] = all_errors.iloc[:, i]

            # Remove any rows with all NaN errors (failed computations)
            valid_rows = ~err_df[vec_order].isna().all(axis=1)
            if not valid_rows.all():
                print(f"Warning: Removed {(~valid_rows).sum()} rows with failed error computations")
                err_df = err_df[valid_rows].copy()

            print(f"Successfully computed errors for {len(err_df)} test cases")
            return err_df
            
    except Exception as e:
        raise RuntimeError(f"Failed to compute error dataframe from {h5_path}: {e}") from e
