import numpy as np
from fmukf.ML.models import MyTimesSeriesTransformer
from fmukf.estimators.UKF import UnscentedKalmanFilter
from fmukf.simulation.container import ScaledContainer
from fmukf.simulation.sensors import SensorModelBase, MaskedFuzzyStateSensor

class FoundationModelUnscentedKalmanFilter(UnscentedKalmanFilter):
    """
    Foundation Model Unscented Kalman Filter (FM-UKF) implementation.
    
    This class extends the standard UKF by replacing the analytical system dynamics
    with a learned neural network model (MyTimesSeriesTransformer). The foundation
    model is used to predict the next state given the current state and control input,
    enabling the UKF to work with complex, learned dynamics instead of analytical models.
    
    Key features:
    - Uses a transformer-based neural network for state propagation
    
    The foundation model is expected to be trained on trajectory data and should
    predict the next state given a sequence of states and control inputs.
    """
    
    def __init__(self,
                 EnvSim: 'ScaledContainer',
                 Sensor: 'MaskedFuzzyStateSensor',
                 xhat0: np.ndarray,
                 P0: np.ndarray,
                 Q: np.ndarray,
                 h: float,
                 model: 'MyTimesSeriesTransformer',
                 max_context: int = 128,
                 integrator_method: str | None = None,
                 sigma_kwargs: str | dict = "Base",
                 recompute_sigma_after_predict: bool = False):
        """
        Initialize the Foundation Model Unscented Kalman Filter.
        
        Parameters
        ----------
        EnvSim: ScaledContainer
            Environment simulation object. Must be a ScaledContainer instance
            as the foundation model is trained on scaled state representations.
            
        Sensor: MaskedFuzzyStateSensor
            Sensor model object. Must be a MaskedFuzzyStateSensor instance
            as the foundation model is only tested with this sensor type.
            
        xhat0: np.ndarray, shape (n_states,)
            Initial state estimate vector.
            
        P0: np.ndarray, shape (n_states, n_states)
            Initial state covariance matrix.
            
        Q: np.ndarray, shape (n_states, n_states)
            Process noise covariance matrix.
            
        h: float
            Time step size for simulation. Must match the foundation model's
            time step (model.h).
            
        model: MyTimesSeriesTransformer
            Trained foundation model that predicts the next state given
            a sequence of states and control inputs. The model should be
            trained on trajectory data from the same environment.
            
        max_context: int, default 128
            Maximum number of previous time steps to use as context for
            the foundation model. Longer contexts provide more information
            but increase computational cost.
            
        integrator_method: str | None, default None
            Integration method for position and heading angle correction.
            If None, no integration is applied. Available methods depend
            on the integrator implementation.
            
        sigma_kwargs: str | dict, default "Base"
            Sigma point generation parameters. If "Base", uses default parameters:
            {"alpha": 1.0, "beta": 0, "kappa": "d"}. Can also be a custom dictionary
            with keys "alpha", "beta", "kappa".
            
        recompute_sigma_after_predict: bool, default False
            Whether to recompute sigma points after the prediction step using
            the predicted mean and covariance. This can improve accuracy but
            increases computational cost.
        
        """
        # Validate input types
        if not isinstance(EnvSim, ScaledContainer):
            raise ValueError(f"EnvSim must be a ScaledContainer instance, got {type(EnvSim)}")
        
        if not isinstance(Sensor, MaskedFuzzyStateSensor):
            raise ValueError(f"Sensor must be a MaskedFuzzyStateSensor instance, got {type(Sensor)}")
        
        super().__init__(EnvSim=EnvSim,
                         Sensor=Sensor,
                         xhat0=xhat0, 
                         P0=P0,
                         Q=Q,
                         h=h,
                         sigma_kwargs=sigma_kwargs,
                         recompute_sigma_after_predict=recompute_sigma_after_predict)
        
        # Override the analytical dynamics with None since we use the foundation model
        self.f = None
        
        # Verify time step consistency
        assert model.h == h, f"model.h={model.h} IS !=  to h={h}. THE LEARNED DYNAMICS WILL PROBABLY BE WRONG!"

        # Save the foundation model and configuration
        self.model: 'MyTimesSeriesTransformer' = model
        self.integrator_method: str | None = integrator_method
        self.max_context: int = max_context

        # Initialize trajectory history for sigma points
        # This stores the full trajectory for each sigma point to provide context
        # to the foundation model
        self.history_: dict[str, np.ndarray] | None = None
    
    def propagate_xpred(self, 
                       points: np.ndarray, 
                       u: np.ndarray) -> np.ndarray:
        """
        Propagate sigma points through the foundation model dynamics.
        
        This method overrides the standard UKF's analytical dynamics with
        the learned foundation model. Instead of using a simple function f(x,u),
        it maintains trajectory history for each sigma point and uses the
        foundation model to predict the next state given the full context.
    
        
        Parameters
        ----------
        points: np.ndarray, shape (n_points, n_states)
            Sigma points to propagate through the foundation model dynamics.
            Each point represents a different state estimate.
            
        u: np.ndarray, shape (n_controls,)
            Control input vector for the current time step.
            
        Returns
        -------
        np.ndarray, shape (n_points, n_states)
            Propagated sigma points predicted by the foundation model.
            Each point represents the predicted next state for the corresponding
            input sigma point.
            
        Notes
        -----
        The foundation model expects input in the format:
        - x: shape (batch_size, seq_len, n_states) - trajectory of states
        - u: shape (batch_size, seq_len, n_controls) - trajectory of controls
        
        The model's unroll method handles the actual prediction by:
        1. Encoding the input trajectories
        2. Processing through the transformer architecture
        3. Decoding the output to get the next state prediction
        4. Optionally applying integration corrections for positions/angles
        """
        
        # Reshape points for batch processing
        # points_: shape (n_points, 1, n_states) - each sigma point as a single timestep
        points_: np.ndarray = points[:,None,:] 
        # u_: shape (1, n_controls) - control input expanded for batch processing
        u_: np.ndarray = u[None,:]        
        
        # Maintain trajectory history for each sigma point
        if self.history_ is None:
            # Initialize history on first call
            self.history_ = {"points": points_, "u": u_}
        else:
            # Append new sigma points to existing trajectory
            # This builds up the trajectory context for the foundation model
            self.history_["points"] = np.concatenate((self.history_["points"], points_), axis=1)   # Shape (b, l, d)
            self.history_["u"]      = np.concatenate((self.history_["u"], u_), axis=0)             # Shape (l, d)

        # Use the foundation model to predict next states
        # The model's unroll method handles the neural network prediction
        X_: np.ndarray = self.model.unroll(x = self.history_["points"],
                                           u = self.history_["u"],
                                           max_context = self.max_context,
                                           integrator_method = self.integrator_method)
        
        # Extract the last predicted state for each sigma point
        # This gives us the next state prediction for each sigma point
        xhat: np.ndarray = X_[:,-1,:]
        return xhat
