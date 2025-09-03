import numpy as np
from omegaconf import DictConfig, OmegaConf
from fmukf.ML.models import MyTimesSeriesTransformer
from fmukf.ML.dataloader import H5LightningDataModule
from fmukf.simulation.container import ScaledContainer, ConstantVelocityContainer
from fmukf.simulation.envsimbase import EnvSimBase
from fmukf.simulation.sensors import SensorModelBase
from fmukf.estimators.UKF import UnscentedKalmanFilter
from fmukf.estimators.End2End import End2EndEstimator, End2EndTransformer
from fmukf.estimators.FMUKF import FoundationModelUnscentedKalmanFilter


def get_device(device_config: str) -> str:
    """
    Determine the device to use for model loading.
    
    Args:
        device_config: Device configuration from config ("cpu", "cuda", "auto")
        
    Returns:
        Device string for torch model loading
    """
    if device_config == "cpu":
        return "cpu"
    elif device_config == "cuda":
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            else:
                print("WARNING: CUDA requested but not available, falling back to CPU")
                return "cpu"
        except ImportError:
            print("WARNING: PyTorch not available, falling back to CPU")
            return "cpu"
    elif device_config == "auto":
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        except ImportError:
            return "cpu"
    else:
        print(f"WARNING: Unknown device config '{device_config}', falling back to CPU")
        return "cpu"




#######################################
# ESTIMATORS
#######################################
class EstimatorWrapper:
    """Wrapper class for estimator methods in the benchmarking system.
    
    This class provides a standardized interface for implementing and initializing different 
    state estimation algorithms. Each estimator is constructed from the hydra/omegaconf config file,
    and implements an estimate() method for processing measurement data for each trajcetory.
    
    The benchmark system creates one instance of each configured estimator type, which are then
    used to process multiple trajectories. Configuration for each estimator is defined in 
    Experiments/config/benchmark.yaml under the "estimators" list, where each entry requires:
    
    1. key: A unique identifier string for the estimator
    2. type: The estimator type (must match one of the types handled in initEstimators())
    3. Various parameters specific to the estimator type (e.g., Q_fac, P0_fac)
    
    Subclasses must implement the estimate() method to process trajectories and return state estimates.
    """
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    def estimate(
        self,
        env: 'ScaledContainer',
        sensor: 'SensorModelBase',
        Y: np.ndarray,
        U: np.ndarray,
        X: np.ndarray,
        h: float
    ) -> np.ndarray:
        """
        Process trajectory data and return state estimates for each time step.

        This method is called for each trajectory in the benchmark system. It receives the environment model,
        sensor model, measurement data, control inputs, and ground truth state (for initialization),
        and should return the estimator's predictions/state estimates.

        Args:
            env: 'ScaledContainer'
                The environment/ship model for this trajectory
            sensor: 'SensorModelBase'
                The sensor model that generated the measurements
            Y: np.ndarray
                Measurement data array with shape (num_time_steps, num_sensor_outputs)
            U: np.ndarray
                Control input array with shape (num_time_steps, num_control_inputs)
            X: np.ndarray
                Ground truth state array with shape (num_time_steps, num_state_variables)
                The first element X[0] is used as the initial state
            h: float
                Time step size in seconds between consecutive measurements
        Returns:
            np.ndarray: State estimates with shape matching X (num_time_steps, num_state_variables)
        """
        raise NotImplementedError


def initEstimators(cfg: DictConfig) -> dict[str, EstimatorWrapper]:
    """Initialize estimator objects from the benchmark configuration.
    
    This function creates instances of all estimator types defined in the config file.
    It is called once during benchmark initialization, and the resulting estimator objects 
    are used for all trajectories in the benchmark.
    
    The benchmark.yaml file defines an "estimators" list where each entry specifies:
    - key: A unique identifier for the estimator instance
    - type: The estimator class to instantiate (e.g., "OracleUKF", "FMUKF", "BaseUKF")
    - Additional parameters specific to each estimator type
    
    Args:
        cfg: 
            DictConfig object containing the complete benchmark configuration
            with an "estimators" list defining all estimator instances to create
    
    Returns:
        Dictionary mapping estimator keys to initialized EstimatorWrapper instances
    
    Raises:
        ValueError: If an invalid estimator type is specified in the configuration
    """

    estimators = {}
    for estimator_cfg in cfg.estimators:
        if estimator_cfg.type == "OracleUKF":
            estimators[estimator_cfg.key] = OracleUKF(cfg, estimator_cfg)
        elif estimator_cfg.type == "BaseUKF":
            estimators[estimator_cfg.key] = BaseUKF(cfg, estimator_cfg)
        elif estimator_cfg.type == "FMUKF":
            estimators[estimator_cfg.key] = FMUKF(cfg, estimator_cfg)
        elif estimator_cfg.type == "CVUKF":
            estimators[estimator_cfg.key] = CVUKF(cfg, estimator_cfg)
        elif estimator_cfg.type == "End2End":
            estimators[estimator_cfg.key] = End2End(cfg, estimator_cfg)
        else:
            raise ValueError(f"Invalid estimator type {estimator_cfg.type}")
    return estimators

class FMUKF(EstimatorWrapper):
    """FMUKF Estimator."""
    def __init__(self, cfg: DictConfig, estimator_cfg: DictConfig):
        super().__init__(cfg)
        assert estimator_cfg.type == "FMUKF"
        self.key = estimator_cfg.key
        self.env = ScaledContainer()
        
        # Remember Process Noise Q and Initial State Covariance 
        self.Q = np.diag([estimator_cfg.Q_fac * ( cfg.sensor_stds[var_name])**2 for var_name in self.env.state_vec_order])
        self.P0 = np.diag([estimator_cfg.P0_fac * ( cfg.sensor_stds[var_name])**2 for var_name in self.env.state_vec_order])
        self.estimator_cfg = estimator_cfg

        # Load the Transformer Model
        device = get_device(estimator_cfg.get("device", "cpu"))
        print(f"Loading FMUKF model on device: {device}")
        self.model = MyTimesSeriesTransformer.load_from_checkpoint(estimator_cfg.ckpt_path, strict=False)#, map_location=device)

    def estimate(
        self,
        env: 'ScaledContainer',
        sensor: 'EnvSimBase',
        Y: np.ndarray,
        U: np.ndarray,
        X: np.ndarray,
        h: float
    ) -> np.ndarray:

        xhat0 = X[0]
        fmukf = FoundationModelUnscentedKalmanFilter(
            EnvSim=env,
            Sensor=sensor,
            xhat0=xhat0,
            P0=self.P0,
            Q=self.Q,
            h=h,
            sigma_kwargs= dict(self.estimator_cfg.sigma_kwargs),
            recompute_sigma_after_predict = self.estimator_cfg.recompute_sigma_after_predict,
            model = self.model,
            max_context = self.estimator_cfg.max_context,
            integrator_method = self.estimator_cfg.integrator_method,
        )
        Xhat = np.array([fmukf(y, u) for y, u in zip(Y, U)])
        return Xhat

class OracleUKF(EstimatorWrapper):
    """OracleUKF Estimator (ie UKF with the correct model)"""
    def __init__(self, cfg: DictConfig, estimator_cfg: DictConfig):
        super().__init__(cfg)
        assert estimator_cfg.type == "OracleUKF"
        self.key = estimator_cfg.key
        self.env = ScaledContainer()
        self.Q = np.diag([estimator_cfg.Q_fac * (cfg.sensor_stds[var_name])**2 for var_name in self.env.state_vec_order])
        self.P0 = np.diag([estimator_cfg.P0_fac * ( cfg.sensor_stds[var_name])**2 for var_name in self.env.state_vec_order])
        self.estimator_cfg = estimator_cfg

    def estimate(
        self,
        env: 'ScaledContainer',
        sensor: 'EnvSimBase',
        Y: np.ndarray,
        U: np.ndarray,
        X: np.ndarray,
        h: float
    ):
        xhat0 = X[0]
        ukf = UnscentedKalmanFilter(
            EnvSim=env,
            Sensor=sensor,
            xhat0=xhat0,
            P0=self.P0,
            Q=self.Q,
            h=h,
            recompute_sigma_after_predict= self.estimator_cfg.recompute_sigma_after_predict,
            sigma_kwargs= dict(self.estimator_cfg.sigma_kwargs),
        )
        Xhat = np.array([ukf(y, u) for y, u in zip(Y, U)])
        return Xhat
    
class BaseUKF(EstimatorWrapper):
    """BaseUKF Estimator (ie UKF always using the base model instead of the correct model)"""
    def __init__(self, cfg: DictConfig, estimator_cfg: DictConfig):
        super().__init__(cfg)
        assert estimator_cfg.type == "BaseUKF"
        self.key = estimator_cfg.key
        self.env = ScaledContainer()
        self.Q = np.diag([estimator_cfg.Q_fac * (cfg.sensor_stds[var_name])**2 for var_name in self.env.state_vec_order])
        self.P0 = np.diag([estimator_cfg.P0_fac * (cfg.sensor_stds[var_name])**2 for var_name in self.env.state_vec_order])
        self.estimator_cfg = estimator_cfg

    def estimate(
        self,
        env: 'ScaledContainer',
        sensor: 'EnvSimBase',
        Y: np.ndarray,
        U: np.ndarray,
        X: np.ndarray,
        h: float
        )   :
        xhat0 = X[0]
        ukf = UnscentedKalmanFilter(
            EnvSim=self.env,
            Sensor=sensor,
            xhat0=xhat0,
            P0=self.P0,
            Q=self.Q,
            h=h,
            recompute_sigma_after_predict= self.estimator_cfg.recompute_sigma_after_predict,
            sigma_kwargs= dict(self.estimator_cfg.sigma_kwargs),
        )
        Xhat = np.array([ukf(y, u) for y, u in zip(Y, U)])
        return Xhat
    
class CVUKF(EstimatorWrapper):
    """CVUKF Estimator (ie UKF with a constant velocity model)"""
    def __init__(self, cfg: DictConfig, estimator_cfg: DictConfig):
        super().__init__(cfg)
        assert estimator_cfg.type == "CVUKF"
        self.key = estimator_cfg.key
        self.env = ConstantVelocityContainer()
        self.Q = np.diag([estimator_cfg.Q_fac * (cfg.sensor_stds[var_name])**2 for var_name in self.env.state_vec_order])
        self.P0 = np.diag([estimator_cfg.P0_fac * (cfg.sensor_stds[var_name])**2 for var_name in self.env.state_vec_order])
        self.estimator_cfg = estimator_cfg

    def estimate(
        self,
        env: 'ScaledContainer',
        sensor: 'EnvSimBase',
        Y: np.ndarray,
        U: np.ndarray,
        X: np.ndarray,
        h: float
    ):
        xhat0 = X[0]
        ukf = UnscentedKalmanFilter(
            EnvSim=self.env,
            Sensor=sensor,
            xhat0=xhat0,
            P0=self.P0,
            Q=self.Q,
            h=h,
            recompute_sigma_after_predict= self.estimator_cfg.recompute_sigma_after_predict,
            # wrap_angles= self.estimator_cfg.wrap_angles,
            sigma_kwargs= dict(self.estimator_cfg.sigma_kwargs),
        )
        Xhat = np.array([ukf(y, u) for y, u in zip(Y, U)])
        return Xhat

class End2End(EstimatorWrapper):
    """End2End Estimator"""
    def __init__(self, cfg: DictConfig, estimator_cfg: DictConfig):
        super().__init__(cfg)
        assert estimator_cfg.type == "End2End"

        device = get_device(estimator_cfg.get("device", "cpu"))
        print(f"Loading End2End model on device: {device}")
        self.model = End2EndTransformer.load_from_checkpoint(estimator_cfg.ckpt_path, strict=False)#, map_location=device)
        self.max_context = estimator_cfg.max_context
        self.integrator_method = estimator_cfg.integrator_method

    def estimate(
        self,
        env: 'ScaledContainer',
        sensor: 'EnvSimBase',
        Y: np.ndarray,
        U: np.ndarray,
        X: np.ndarray,
        h: float
    ):
        # Create a single End2EndEstimator instanc
        e2e = End2EndEstimator(self.model,
                               sensor,
                               h=h,
                               max_context=self.max_context,
                               integrator_method=self.integrator_method)
        
        # Call it sequentially for each (y, u) pair to maintain state
        Xhat = []
        for y, u in zip(Y, U):
            xhat = e2e(y, u)
            Xhat.append(xhat)
        return np.array(Xhat)

