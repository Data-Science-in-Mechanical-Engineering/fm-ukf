
import torch
import lightning as L
import h5py
import numpy as np
import yaml
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

class H5LightningDataModule(L.LightningDataModule):
    """
    PyTorch Lightning DataModule for loading time series data from HDF5 files.
    
    This module handles loading, splitting, and preprocessing of trajectory data stored in HDF5 format.
    It automatically computes and caches normalization statistics, handles train/validation/test splits
    based on environment keys, and provides DataLoaders for PyTorch Lightning training.
    
    The HDF5 file structure is expected to be:
    - Root level: Environment keys (e.g., "env_0", "env_1", ...)
    - Each environment contains trajectory keys (e.g., "traj_0", "traj_1", ...)
    - Each trajectory contains:
        - "x": State data of shape [time_steps, state_dim]  
        - "u": Control data of shape [time_steps, control_dim]
    
    Key Features:
    1. **Deterministic Splits**: Uses seed to ensure reproducible train/val/test splits
    2. **Statistics Caching**: Computes and caches normalization statistics in the HDF5 file
    3. **Sequence Truncation**: Optionally truncates long trajectories to a maximum length
    4. **Flexible Ordering**: Configurable state and control vector feature ordering
    
    Args:
        h5_file_path_str (str):
            Path to the HDF5 file containing trajectory data
        num_envs_val (int):
            Number of environments to allocate to validation set
        num_envs_test (int):
            Number of environments to allocate to test set
        seed (int):
            Random seed for reproducible train/validation/test splits
        batch_size (int, optional):
            Batch size for DataLoaders. Defaults to 32.
        dtype (torch.dtype, optional):
            Data type for loaded tensors. Defaults to torch.float32.
        seq_len (int | None, optional):
            Maximum sequence length. Longer trajectories are truncated from the start.
            If None, no truncation is applied. Defaults to None.
        compute_statistics (bool, optional):
            Whether to compute mean and std statistics for normalization. Defaults to True.
        recompute_statistics (bool, optional):
            Whether to recompute statistics even if cached versions exist. Defaults to False.
        state_vec_order (list[str] | None, optional):
            Order of features in the state vector. If None, uses default ordering:
            ["u", "v", "r", "x", "y", "psi", "p", "phi", "delta", "n"]. Defaults to None.
        input_vec_order (list[str] | None, optional):
            Order of features in the control vector. If None, uses default ordering:
            ["delta", "n"]. Defaults to None.
        silent (bool, optional):
            Whether to suppress verbose output during loading. Defaults to True.
    
    Example:
        >>> datamodule = H5LightningDataModule(
        ...     h5_file_path_str="data/trajectories.h5",
        ...     num_envs_val=10,
        ...     num_envs_test=20,
        ...     seed=69,
        ...     batch_size=64
        ... )
        >>> datamodule.setup("fit")
        >>> train_loader = datamodule.train_dataloader()
        >>> val_loader = datamodule.val_dataloader()
    """
    
    def __init__(
        self,
        h5_file_path_str: str,
        num_envs_val: int,
        num_envs_test: int,
        seed: int,
        batch_size: int = 32,
        dtype: torch.dtype = torch.float32,
        seq_len: int | None = None,  # optional sequence length
        compute_statistics: bool = True,
        recompute_statistics: bool = False,
        state_vec_order: list[str] | None = None,
        input_vec_order: list[str] | None = None,
        silent: bool = True, # Prints less stuff
    ):
        super().__init__()
        self.h5_file_path_str = h5_file_path_str
        self.num_envs_val = num_envs_val
        self.num_envs_test = num_envs_test
        self.seed = seed
        self.batch_size = batch_size
        self.dtype = dtype
        self.seq_len = seq_len
        self.silent = silent

        # Set orders with defaults if not provided.
        self.state_vec_order = state_vec_order if state_vec_order is not None else \
            ["u", "v", "r", "x", "y", "psi", "p", "phi", "delta", "n"]
        self.input_vec_order = input_vec_order if input_vec_order is not None else \
            ["delta", "n"]

        self.compute_statistics_flag = compute_statistics
        self.recompute_statistics = recompute_statistics

        if self.compute_statistics_flag:
            self.statistics = self.compute_and_cache_statistics()
        else:
            self.statistics = None

    def compute_and_cache_statistics(self) -> dict:
        """
        Compute normalization statistics for state and control vectors across all trajectories.
        
        This method computes mean and standard deviation for each feature in the state and control
        vectors by concatenating all trajectories and computing statistics over the time dimension.
        Results are cached in the HDF5 file to avoid recomputation on subsequent runs.
        
        Returns:
            dict: Statistics dictionary with structure:
                {
                    "state": {var_name: (mean, std), ...},
                    "input": {var_name: (mean, std), ...}
                }
                
        Note:
            Statistics are cached as a YAML string in the HDF5 file's "statistics" attribute.
            Use recompute_statistics=True to force recomputation of cached statistics.
        """
        # First, try to load existing stats if they exist and we are not recomputing.
        with h5py.File(self.h5_file_path_str, "r") as f:
            if "statistics" in f.attrs and not self.recompute_statistics:
                try:
                    stats = yaml.safe_load(f.attrs["statistics"])
                    print("Loaded cached statistics from file.")
                    return stats
                except Exception as e:
                    print("Failed to parse cached statistics, recomputing:", e)

            # Otherwise, compute statistics.
            state_all = []
            input_all = []
            for env_key in tqdm(f.keys(), desc="Computing mean and statistics over environments", leave=False):
                env_group = f[env_key]
                for traj_key in env_group.keys():
                    traj_group = env_group[traj_key]
                    x = traj_group["x"][:]  # expected shape: (T, 10)
                    u = traj_group["u"][:]  # expected shape: (T, 2)
                    state_all.append(x)
                    input_all.append(u)
            
            # Concatenate all trajectories along time dimension.
            state_all = np.concatenate(state_all, axis=0)  # shape: (total_timesteps, 10)
            input_all = np.concatenate(input_all, axis=0)  # shape: (total_timesteps, 2)

            stats = {"state": {}, "input": {}}
            for i, var in enumerate(self.state_vec_order):
                mean_val = float(np.mean(state_all[:, i]))
                std_val = float(np.std(state_all[:, i]))
                stats["state"][var] = (mean_val, std_val)
            for i, var in enumerate(self.input_vec_order):
                mean_val = float(np.mean(input_all[:, i]))
                std_val = float(np.std(input_all[:, i]))
                stats["input"][var] = (mean_val, std_val)
        
        # Cache the computed statistics in the H5 file.
        try:
            with h5py.File(self.h5_file_path_str, "a") as f:
               f.attrs["statistics"] = yaml.safe_dump(stats)
            print("Cached computed statistics to file.")
        except Exception as e:
            print("Warning: Could not cache statistics:", e)

        return stats

    def setup(self, stage: str | None = None) -> tuple[list[str], list[str], list[str]] | None:
        """
        Setup data splits for PyTorch Lightning training stages.
        
        This method loads and prepares datasets based on the requested stage. It deterministically
        splits environments into train/validation/test sets using the provided seed, then loads
        the corresponding trajectory data into TensorDatasets.
        
        Args:
            stage (str | None, optional):
                Training stage to setup data for:
                - "fit": Load training and validation datasets
                - "validate": Load only validation dataset  
                - "test": Load only test dataset
                - "getKeys": Return environment keys for each split (debugging and used in benchmarker)
                - None: Load all datasets
                Defaults to None.
                
        Returns:
            tuple[list[str], list[str], list[str]] | None:
                If stage == "getKeys", returns (train_keys, val_keys, test_keys).
                Otherwise returns None.
                
        Shape Transformations:
            - Raw trajectory data: x[time_steps, state_dim], u[time_steps, control_dim]
            - After loading: x[batch_size, seq_len, state_dim], u[batch_size, seq_len, control_dim]
            - TensorDataset order: (x, u) - state comes first, then control
        """
        with h5py.File(self.h5_file_path_str, "r") as f:
            # Sort environment keys to ensure consistent ordering
            env_keys = sorted(list(f.keys()))
            rng = np.random.RandomState(self.seed)
            env_keys_shuffled = rng.permutation(env_keys)
            
            # Determine splits based on the number of environments allocated
            val_keys = env_keys_shuffled[: self.num_envs_val]
            test_keys = env_keys_shuffled[self.num_envs_val : self.num_envs_val + self.num_envs_test]
            train_keys = env_keys_shuffled[self.num_envs_val + self.num_envs_test :]

            for key in test_keys:
                assert key not in val_keys
                assert key not in train_keys
            for key in val_keys:
                assert key not in train_keys

            # Print keys for debugging

            if not self.silent:
                print("train_keys:", train_keys)
                print("val_keys:", val_keys)
                print("test_keys:", test_keys)

            def load_samples(keys):
                samples_u = []
                samples_x = []
                for key in tqdm(keys, desc="Loading environment", total=len(keys), leave=False):
                    env_group = f[key]
                    for traj_key in env_group.keys():
                        traj_group = env_group[traj_key]
                        u = traj_group["u"][:]  # load control input
                        x = traj_group["x"][:]  # load state/output
                        # If seq_len is specified and the trajectory is longer, take the last seq_len entries
                        if self.seq_len is not None:
                            if len(u) > self.seq_len:
                                u = u[-self.seq_len:]
                            if len(x) > self.seq_len:
                                x = x[-self.seq_len:]
                        samples_u.append(u)
                        samples_x.append(x)
                return samples_u, samples_x

            if stage == "fit":
                # Load both training and validation splits
                train_u, train_x = load_samples(train_keys)
                val_u, val_x = load_samples(val_keys)
                # Swapped order: x comes first then u
                self.train_dataset = TensorDataset(
                    torch.tensor(np.array(train_x), dtype=self.dtype),
                    torch.tensor(np.array(train_u), dtype=self.dtype)
                )
                self.val_dataset = TensorDataset(
                    torch.tensor(np.array(val_x), dtype=self.dtype),
                    torch.tensor(np.array(val_u), dtype=self.dtype)
                )

            elif stage == "validate":
                # Load only the validation split with swapped order
                val_u, val_x = load_samples(val_keys)
                self.val_dataset = TensorDataset(
                    torch.tensor(np.array(val_x), dtype=self.dtype),
                    torch.tensor(np.array(val_u), dtype=self.dtype)
                )

            elif stage == "test":
                # Load only the test split with swapped order
                test_u, test_x = load_samples(test_keys)
                self.test_dataset = TensorDataset(
                    torch.tensor(np.array(test_x), dtype=self.dtype),
                    torch.tensor(np.array(test_u), dtype=self.dtype)
                )

            elif stage == "getKeys":
                # (This is only useful for debugging, and just getting the keys for the specific split)
                # print("stage is 'getKeys', hence not loading any data")
                return train_keys, val_keys, test_keys

            elif stage is None:
                # Load all splits if no specific stage is provided with swapped order
                train_u, train_x = load_samples(train_keys)
                val_u, val_x = load_samples(val_keys)
                test_u, test_x = load_samples(test_keys)
                self.train_dataset = TensorDataset(
                    torch.tensor(np.array(train_x), dtype=self.dtype),
                    torch.tensor(np.array(train_u), dtype=self.dtype)
                )
                self.val_dataset = TensorDataset(
                    torch.tensor(np.array(val_x), dtype=self.dtype),
                    torch.tensor(np.array(val_u), dtype=self.dtype)
                )
                self.test_dataset = TensorDataset(
                    torch.tensor(np.array(test_x), dtype=self.dtype),
                    torch.tensor(np.array(test_u), dtype=self.dtype)
                )

    def train_dataloader(self) -> DataLoader:
        """
        Create DataLoader for training dataset.
        
        Returns:
            DataLoader:
                Training DataLoader with shuffling enabled and configured batch size
        """
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        """
        Create DataLoader for validation dataset.
        
        Returns:
            DataLoader:
                Validation DataLoader with shuffling disabled and configured batch size
        """
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        """
        Create DataLoader for test dataset.
        
        Returns:
            DataLoader:
                Test DataLoader with shuffling disabled and configured batch size
        """
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)