import numpy as np
from .estimator_base import EstimatorBase

from fmukf.simulation.envsimbase import EnvSimBase
from fmukf.simulation.sensors import SensorModelBase
from fmukf.simulation.container import ScaledContainer

class UnscentedKalmanFilter(EstimatorBase):
    """
    Unscented Kalman Filter (UKF) implementation with angle wrapping support.
    
    This class implements the Unscented Kalman Filter algorithm for state estimation
    in nonlinear systems. It extends the standard UKF with special handling for
    angular variables (like heading angles) to avoid discontinuities at 0°/360°.
    
    The filter uses sigma points to propagate the state distribution through the
    nonlinear system dynamics and measurement functions, providing better accuracy
    than the Extended Kalman Filter for highly nonlinear systems.
    
    Key features:
    - Automatic angle wrapping for angular state variables (e.g., heading angle psi)
    - Circular mean calculation for angular dimensions
    - Configurable sigma point parameters (alpha, beta, kappa)
    - Optional sigma point recomputation after prediction step
    - Comprehensive history tracking for debugging and analysis
    """
    
    def __init__(self,
                 EnvSim: 'EnvSimBase',
                 Sensor: 'SensorModelBase', 
                 xhat0: np.ndarray,
                 P0: np.ndarray, 
                 Q: np.ndarray, 
                 h: float, 
                 sigma_kwargs: str | dict = "Base", 
                 recompute_sigma_after_predict: bool = False,
                 x_angle_dims: list[int] | None = None,
                 y_angle_dims: list[int] | None = None):
        """
        Initialize the Unscented Kalman Filter.
        
        Parameters
        ----------
        EnvSim: EnvSimBase
            Environment simulation object containing the system dynamics.
            Must have sim_step(x, u, h) method and state_vec_order attribute.
            
        Sensor: SensorModelBase
            Sensor model object containing measurement functions.
            Must have g_function(x, u) method, stds attribute, and y_vec_order attribute.
            
        xhat0: np.ndarray, shape (n_states,)
            Initial state estimate vector.
            
        P0: np.ndarray, shape (n_states, n_states)
            Initial state covariance matrix.
            
        Q: np.ndarray, shape (n_states, n_states)
            Process noise covariance matrix.
            
        h: float
            Time step size for simulation.
            
        sigma_kwargs: str | dict, default "Base"
            Sigma point generation parameters. If "Base", uses default parameters:
            {"alpha": 1.0, "beta": 0, "kappa": "d"}. Can also be a custom dictionary
            with keys "alpha", "beta", "kappa".
            
        recompute_sigma_after_predict: bool, default False
            Whether to recompute sigma points after the prediction step using
            the predicted mean and covariance. This can improve accuracy but
            increases computational cost.
            
        x_angle_dims: list[int] | None, default None
            List of state vector dimension indices that should be treated as angles.
            If None, automatically detects "psi" dimension for ScaledContainer,
            otherwise uses empty list.
            
        y_angle_dims: list[int] | None, default None
            List of measurement vector dimension indices that should be treated as angles.
            If None, automatically detects "psi" dimension in sensor measurements,
            otherwise uses empty list.
        """
        # Format Parametres for Sigma Point and Weight Setup
        if sigma_kwargs == "Base":
            sigma_kwargs = dict(alpha = 1.0, beta = 0, kappa = "d")
        self.sigma_kwargs = sigma_kwargs
        self.recompute_sigma_after_predict = recompute_sigma_after_predict
        
        # Transition Function and Process Noise
        self.f = lambda x, u: EnvSim.sim_step(x, u, h)
        self.Q = Q

        # Sensor Model and Noise
        self.g = lambda x, u: Sensor.g_function(x,u)
        self.R = np.diag(Sensor.stds)**2 
                
        # Remember Initial Estimates
        self.xhat_mean = xhat0
        self.xhat_cov = P0
        self.is_initial_step = True  # Flag to indicate first time-step

        # Initialize Lists to Remember History of different Parameters (useful for debugging)
        self.history = {name:[] for name in ["xpred_mean", "xpred_cov", "ypred_mean", "ypred_cov", "xhat_mean", "xhat_cov"]}

        # Add angle wrapping parameters
        self.angle_wrap = 360.0  # Angle wrapping modulus
        
        # Set angle dimensions with logic for ScaledContainer
        if x_angle_dims is not None:
            self.x_angle_dims = x_angle_dims
        else:
            # Default logic: if ScaledContainer, use psi dimension, otherwise empty
            if isinstance(EnvSim, ScaledContainer) and "psi" in EnvSim.state_vec_order:
                self.x_angle_dims = [EnvSim.state_vec_order.index("psi")]
            else:
                self.x_angle_dims = []
                
        if y_angle_dims is not None:
            self.y_angle_dims = y_angle_dims
        else:
            # Default logic: if psi in sensor measurements, use it
            if "psi" in Sensor.y_vec_order:
                self.y_angle_dims = [Sensor.y_vec_order.index("psi")]
            else:
                self.y_angle_dims = []

    def propagate_xpred(self, 
                       points: np.ndarray, 
                       u: np.ndarray) -> np.ndarray:
        """
        Propagate sigma points through the system dynamics with angle wrapping.
        
        Parameters
        ----------
        points: np.ndarray, shape (n_points, n_states)
            Sigma points to propagate through the system dynamics.
            
        u: np.ndarray, shape (n_controls,)
            Control input vector.
            
        Returns
        -------
        np.ndarray, shape (n_points, n_states)
            Propagated sigma points with angles wrapped to [0, 360) range.
        """
        return np.array([self.wrap_angles(self.f(self.wrap_angles(x, self.x_angle_dims), u), self.x_angle_dims) for x in points])

    def __call__(self, 
                 y: np.ndarray, 
                 u: np.ndarray) -> np.ndarray:
        """
        Perform one step of the Unscented Kalman Filter.
        
        This method implements the predict-update cycle of the UKF:
        1. Generate sigma points from current state estimate
        2. Propagate sigma points through system dynamics (predict)
        3. Transform sigma points through measurement function
        4. Update state estimate using measurement (update)
        
        Parameters
        ----------
        y: np.ndarray, shape (n_measurements,)
            Measurement vector. Angular measurements are automatically wrapped
            to [0, 360) range.
            
        u: np.ndarray, shape (n_controls,)
            Control input vector for the current time step.
            
        Returns
        -------
        np.ndarray, shape (n_states,)
            Updated state estimate after the predict-update cycle.
        """
        # Use given parameters if first step

        if len(self.y_angle_dims) != 0:
            y = y.copy()
            y[self.y_angle_dims] = y[self.y_angle_dims] % self.angle_wrap

        if self.is_initial_step:
            self.is_initial_step = False
            # Set history for these parameters with nans (since they are not computed) 
            self.history["xpred_mean"].append(np.full(self.xhat_mean.shape, np.nan))
            self.history["xpred_cov"].append(np.full(self.xhat_cov.shape, np.nan))
            self.history["ypred_mean"].append(np.full(y.shape, np.nan))
            self.history["ypred_cov"].append(np.full((len(y),len(y)), np.nan))
            self.history["xhat_mean"].append(self.xhat_mean)
            self.history["xhat_cov"].append(self.xhat_cov)

            # Remember input for next time-step
            self.u_previous = u
            return self.xhat_mean


        # Compute Sigma Points and Weights (with angle wrapping)
        points, weightsMean, weightsCov = self.SigmaPointWeights(self.xhat_mean, self.xhat_cov, **self.sigma_kwargs)
        
        # Predicted States Distribution (with angle wrapping)
        points_xpred = self.propagate_xpred(points,  self.u_previous)

        # Circular mean calculation for state
        xpred_mean = self.circular_mean(points_xpred, weightsMean, self.x_angle_dims)
        
        # Covariance calculation with angular residuals
        residuals_x = self.angular_residuals(points_xpred, xpred_mean, self.x_angle_dims)
        xpred_cov = np.einsum("k,ki,kj->ij", weightsCov, residuals_x, residuals_x) + self.Q

        if self.recompute_sigma_after_predict:
            points_xpred, weightsMean, weightsCov = self.SigmaPointWeights(xpred_mean, xpred_cov, **self.sigma_kwargs)
            points_xpred = self.wrap_angles(points_xpred, self.x_angle_dims)
            residuals_x = self.angular_residuals(points_xpred, xpred_mean, self.x_angle_dims)

        # Predicted Measurement Distribution (with angle handling)
        points_ypred = np.array([self.g(x, u) for x in points_xpred])
        points_ypred = self.wrap_angles(points_ypred, self.y_angle_dims)
        
        # Circular mean for measurement
        ypred_mean = self.circular_mean(points_ypred, weightsMean, self.y_angle_dims)
        
        # Measurement covariance with angular residuals
        residuals_y = self.angular_residuals(points_ypred, ypred_mean, self.y_angle_dims)
        ypred_cov = np.einsum("k,ki,kj->ij", weightsCov, residuals_y, residuals_y) + self.R

        # Cross-variance with angular residuals
        crossvar_xy = np.einsum("k,ki,kj->ij", weightsCov, residuals_x, residuals_y)

        # Kalman Gain
        K = crossvar_xy @ np.linalg.inv(ypred_cov)

        # Update with angular innovation
        innovation = self.angular_innovation(y, ypred_mean, self.y_angle_dims)
        self.xhat_mean = xpred_mean + K @ innovation
        self.xhat_mean = self.wrap_angles(self.xhat_mean, self.x_angle_dims)  # Wrap final estimate
        self.xhat_cov = xpred_cov - K @ ypred_cov @ K.T

        # Remember input u for next time-step
        self.u_previous = u

        # Log values in history
        self.history["xpred_mean"].append(xpred_mean)
        self.history["xpred_cov"].append(xpred_cov)
        self.history["ypred_mean"].append(ypred_mean)
        self.history["ypred_cov"].append(ypred_cov)
        self.history["xhat_mean"].append(self.xhat_mean)
        self.history["xhat_cov"].append(self.xhat_cov)

        return self.xhat_mean

    def wrap_angles(self, 
                   arr: np.ndarray, 
                   angle_dims: list[int]) -> np.ndarray:
        """
        Wrap angles to [0, 360) range for specified dimensions.
        
        Parameters
        ----------
        arr: np.ndarray, shape (..., n_dims)
            Array containing values to wrap. Can be 1D or multi-dimensional.
            
        angle_dims: list[int]
            List of dimension indices that should be treated as angles
            and wrapped to [0, 360) range.
            
        Returns
        -------
        np.ndarray, shape (..., n_dims)
            Array with angles wrapped to [0, 360) range. Non-angular
            dimensions remain unchanged.
        """
        arr = arr.copy()
        for dim in angle_dims:
            arr[..., dim] = np.mod(arr[..., dim], self.angle_wrap)
        return arr

    def circular_mean(self, 
                     points: np.ndarray, 
                     weights: np.ndarray, 
                     angle_dims: list[int]) -> np.ndarray:
        """
        Calculate circular mean for angular dimensions and linear mean for others.
        
        For angular dimensions, computes the circular mean using trigonometric
        functions to handle the periodicity. For non-angular dimensions,
        computes the standard weighted linear mean.
        
        Parameters
        ----------
        points: np.ndarray, shape (n_points, n_dims)
            Array of points to compute mean from.
            
        weights: np.ndarray, shape (n_points,)
            Weights for each point in the mean calculation.
            
        angle_dims: list[int]
            List of dimension indices that should be treated as angles
            and use circular mean calculation.
            
        Returns
        -------
        np.ndarray, shape (n_dims,)
            Mean vector with circular mean for angular dimensions and
            linear mean for non-angular dimensions.
        """
        mean = np.einsum("k,ki->i", weights, points)  # Linear mean
        for dim in angle_dims:
            angles = np.deg2rad(points[:, dim])
            sum_sin = np.dot(weights, np.sin(angles))
            sum_cos = np.dot(weights, np.cos(angles))
            circ_mean_rad = np.arctan2(sum_sin, sum_cos)
            mean[dim] = np.rad2deg(circ_mean_rad) % self.angle_wrap
        return mean

    def angular_residuals(self, 
                         points: np.ndarray, 
                         mean: np.ndarray, 
                         angle_dims: list[int]) -> np.ndarray:
        """
        Calculate residuals with angular wrapping for specified dimensions.
        
        For angular dimensions, computes the shortest angular distance between
        points and mean. For non-angular dimensions, computes standard residuals.
        
        Parameters
        ----------
        points: np.ndarray, shape (n_points, n_dims)
            Array of points to compute residuals from.
            
        mean: np.ndarray, shape (n_dims,)
            Mean vector to compute residuals relative to.
            
        angle_dims: list[int]
            List of dimension indices that should be treated as angles
            and use angular residual calculation.
            
        Returns
        -------
        np.ndarray, shape (n_points, n_dims)
            Residuals array with angular residuals for angular dimensions
            and linear residuals for non-angular dimensions.
        """
        residuals = points - mean
        for dim in angle_dims:
            diff = points[:, dim] - mean[dim]
            residuals[:, dim] = (diff + self.angle_wrap/2) % self.angle_wrap - self.angle_wrap/2
        return residuals

    def angular_innovation(self, 
                          measurement: np.ndarray, 
                          prediction: np.ndarray, 
                          angle_dims: list[int]) -> np.ndarray:
        """
        Calculate innovation with angular wrapping for specified dimensions.
        
        The innovation is the difference between measurement and prediction.
        For angular dimensions, computes the shortest angular distance.
        For non-angular dimensions, computes standard difference.
        
        Parameters
        ----------
        measurement: np.ndarray, shape (n_measurements,)
            Measurement vector.
            
        prediction: np.ndarray, shape (n_measurements,)
            Predicted measurement vector.
            
        angle_dims: list[int]
            List of dimension indices that should be treated as angles
            and use angular innovation calculation.
            
        Returns
        -------
        np.ndarray, shape (n_measurements,)
            Innovation vector with angular innovation for angular dimensions
            and linear innovation for non-angular dimensions.
        """
        innovation = measurement - prediction
        for dim in angle_dims:
            diff = measurement[dim] - prediction[dim]
            innovation[dim] = (diff + self.angle_wrap/2) % self.angle_wrap - self.angle_wrap/2
        return innovation

    def SigmaPointWeights(self,
                          mean: np.ndarray,
                          Cov: np.ndarray,
                          alpha: float = 1.0,
                          beta: float = 0,
                          kappa: str | float = "3d+1",
                          use_SVD: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate sigma points and weights for the Unscented Transform.
        
        This method implements the standard UKF sigma point generation algorithm.
        Sigma points are generated around the mean using the Cholesky decomposition
        or SVD of the covariance matrix.
        
        Parameters
        ----------
        mean: np.ndarray, shape (n_dims,)
            Mean vector around which to generate sigma points.
            
        Cov: np.ndarray, shape (n_dims, n_dims)
            Covariance matrix for sigma point generation.
            
        alpha: float, default 1.0
            Scaling parameter that determines the spread of sigma points.
            Typically set to 1e-3 for Gaussian distributions.
            
        beta: float, default 0
            Parameter to incorporate prior knowledge of the distribution.
            beta = 2 is optimal for Gaussian distributions.
            
        kappa: str | float, default "3d+1"
            Secondary scaling parameter. Can be:
            - "3d+1": kappa = 3*n_dims + 1 (default)
            - "d": kappa = n_dims
            - float: custom value
            
        use_SVD: bool, default True
            Whether to use SVD decomposition (True) or Cholesky decomposition (False)
            for matrix factorization. SVD is more numerically stable.
            
        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            - SigmaPoints: np.ndarray, shape (2*n_dims + 1, n_dims)
                Generated sigma points
            - weightsMean: np.ndarray, shape (2*n_dims + 1,)
                Weights for mean calculation
            - weightsCov: np.ndarray, shape (2*n_dims + 1,)
                Weights for covariance calculation
        """
        # Following the notation from here https://en.wikipedia.org/wiki/Kalman_filter#Unscented_Kalman_filter
        d = len(mean)
        N = 2*d + 1 # Number of Sigma points
        if kappa == "3d+1":
            kappa = 3*d+1
        elif kappa == "d":
            kappa = d

        # Mean Weights
        weightsMean = np.ones((N)) * 1 / (2 * kappa * alpha**2 )
        weightsMean[0] = (kappa * alpha**2 - d) / (kappa * alpha**2)

        # Covariance Weights
        weightsCov = np.ones((N)) * 1 / (2 * kappa * alpha**2)
        weightsCov[0] = weightsMean[0] + 1 - alpha**2 + beta

        # Compute A s.t. Cov = A @ A.T
        if not use_SVD:
            A = np.linalg.cholesky(Cov)
        else:
            U,S,_ = np.linalg.svd(Cov)
            A = U @ np.diag(np.sqrt(S))
        
        # Sigma Points
        SigmaPoints = np.zeros((N, d))
        SigmaPoints[0] = mean
        SigmaPoints[1:1+d] = mean + alpha * np.sqrt(kappa) * A.T
        SigmaPoints[1+d:]  = mean - alpha * np.sqrt(kappa) * A.T
        return SigmaPoints, weightsMean, weightsCov


import holoviews as hv

def VisualizeKalman(EnvSim: 'EnvSimBase',
                   Estimator: 'UnscentedKalmanFilter',
                   Sensor: 'SensorModelBase',
                   trajdict: dict,
                   alpha_spread: float = 0.4,
                   alpha_spread_muted: float = 0.05):
    """
    Create interactive visualization of Kalman filter performance.
    
    This function generates a comprehensive visualization of the Kalman filter
    performance, showing ground truth, measurements, predictions, and estimates
    for all state variables. The visualization includes uncertainty bands and
    allows interactive exploration of the results.
    
    Parameters
    ----------
    EnvSim: EnvSimBase
        Environment simulation object containing state_vec_order attribute.
        
    Estimator: UnscentedKalmanFilter
        Kalman filter estimator object containing history attribute with
        keys: ["xpred_mean", "xpred_cov", "ypred_mean", "ypred_cov", "xhat_mean", "xhat_cov"]
        
    Sensor: SensorModelBase
        Sensor model object containing y_vec_order attribute.
        
    trajdict: dict
        Dictionary containing trajectory data with keys:
        - "t": np.ndarray, shape (n_timesteps,) - Time vector
        - "x": np.ndarray, shape (n_timesteps, n_states) - Ground truth states
        - "y": np.ndarray, shape (n_timesteps, n_measurements) - Measurements
        
    alpha_spread: float, default 0.4
        Alpha value for uncertainty band visualization (0-1).
        
    alpha_spread_muted: float, default 0.05
        Alpha value for muted uncertainty bands (0-1).
        
    Returns
    -------
    holoviews.NdLayout
        Interactive visualization layout with subplots for each state variable.
        Each subplot shows:
        - Blue line: Ground truth
        - Green scatter: Measurements (if available)
        - Green bands: Measurement uncertainty
        - Orange line/bands: Predicted state and uncertainty
        - Red line/bands: Estimated state and uncertainty
    """
    #TODO: Make argument variable names consistentfor Estimator, Sensor, EnvSim,trajdict consistent with the rest of the codebase
    plotdict = {} # <-- Save subplot for every variable in here
    history = Estimator.history
    t = trajdict["t"]

    # Subplot everything for every variable
    for idx, varname in list(enumerate(EnvSim.state_vec_order))[::-1]:
        # Ground Truth (always plotted)
        xtrue = trajdict["x"][:, idx]
        subplot = hv.Curve((t,xtrue), label="x true").opts(line_color="blue")

        # Sensor Measurements and Estimates
        if varname in Sensor.y_vec_order:
            idx_y = Sensor.y_vec_order.index(varname)

            # True measurments
            y = trajdict["y"][:, idx_y]
            subplot = hv.Scatter(zip(t,y), label = "y true").opts(color='green') * subplot

            # Predicted Measurements and Uncertainty
            ypred = np.array(history["ypred_mean"])[:, idx_y]
            ypred_std = np.sqrt(np.array(history["ypred_cov"])[:, idx_y, idx_y])
            subplot = subplot * hv.Spread((t, ypred, ypred_std), label = "y std").opts(fill_color='green', fill_alpha=alpha_spread, muted_alpha=alpha_spread_muted, line_color=None)

        # Post update Estimate and Uncertainty
        xhat = np.array(history["xhat_mean"])[:, idx]
        xhat_std = np.sqrt(np.array(history["xhat_cov"])[:, idx, idx])
        subplot = hv.Spread((t, xhat, xhat_std), label = "xhat std").opts(fill_color='red', fill_alpha=alpha_spread, muted_alpha=alpha_spread_muted, line_color=None) * subplot
        subplot = hv.Curve((t, xhat), label="xhat").opts(line_color="red")  * subplot

        # Pre update Estimate and uncertainty
        xpred = np.array(history["xpred_mean"])[:, idx]
        xpred_std = np.sqrt(np.array(history["xpred_cov"])[:, idx, idx])
        subplot = hv.Spread((t, xpred, xpred_std), label = "xpred std").opts(fill_color='orange', fill_alpha=alpha_spread, muted_alpha=alpha_spread_muted, line_color=None) * subplot
        subplot = hv.Curve((t,xpred), label="xpred").opts(line_color="orange") * subplot

        # Format Variable
        subplot.opts(xlabel="t / s", ylabel=varname, width=500)
        plotdict[varname] = subplot

    return hv.NdLayout(plotdict).opts( shared_axes=False)