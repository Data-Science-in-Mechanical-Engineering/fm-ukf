import torch
torch.set_default_dtype(torch.float32)
from torch import Tensor
import numpy as np
import fmukf

from fmukf.ML.models import MyTimesSeriesTransformer
from .estimator_base import EstimatorBase
from fmukf.simulation.sensors import MaskedFuzzyStateSensor

class End2EndTransformer(MyTimesSeriesTransformer):        
    """
    End-to-end transformer for state estimation specifically designed for MaskedFuzzySensor.
    
    This model extends MyTimesSeriesTransformer to handle sensor-based state estimation tasks,
    but is **specifically designed to work only with MaskedFuzzySensor**. This design choice
    enables maximum reuse of the existing transformer architecture and training pipeline.
    
    **Why MaskedFuzzySensor Only?**
    MaskedFuzzySensor produces observations y_k = sensor(x_k) where:
    - y_k contains a **subset** of state features from x_k (some features are masked/missing)
    - Observed features have **additive Gaussian noise** applied
    - Missing features are simply **not present** in the observation vector
    
    This structure allows us to reuse MyTimesSeriesTransformer's functionality by:
    1. **Feature Transformations**: Same encode_x/decode_x (psi → sin/cos, normalization)
    2. **Noise Handling**: Same min_std_multiplier/max_std_multiplier noise injection
    3. **Architecture**: Identical transformer architecture (patching, attention, etc.)
    4. **Training Pipeline**: Same loss functions, optimizers, learning rate schedules
    
    **Model Workflow:**
    The model learns: ((y_0, u_0), ..., (y_l, u_l)) → (x_0, ..., x_l)
    
    During training:
    1. Start with true states x and controls u
    2. Simulate MaskedFuzzySensor by randomly selecting a sensor configuration
    3. Apply noise to x, then mask out unobserved features → creates y
    4. Concatenate [y, u] as input, use true encoded x as target
    5. Train transformer to reconstruct full state from partial observations
    
    **Key Reused Components from MyTimesSeriesTransformer:**
    - encode_x/decode_x: Handles psi angle transformation and normalization
    - noise_std_x/noise_std_u: Training noise injection parameters  
    - Transformer architecture: Patching, attention layers, loss functions
    - Training hyperparameters: Learning rates, optimizers, schedules
    - MLHP is disabled since masking is handled by sensor configurations
    
    Args:
        sensor_configs (list[list[str]]):
            List of MaskedFuzzySensor configurations. Each configuration specifies
            which state features are observed (unmasked). Features not in the list
            are masked as zero during training.
        *args:
            Additional positional arguments passed to MyTimesSeriesTransformer
        **kwargs:
            Additional keyword arguments passed to MyTimesSeriesTransformer
            
    Example Sensor Configurations:
        >>> sensor_configs = [
        ...     ["u", "v", "r", "x", "y", "psi", "p", "phi", "delta", "n"],   # Full state
        ...     ["r", "x", "y", "psi"],                                       # IMU + GPS
        ...     ["u", "v", "r"],                                              # IMU only  
        ... ]
        
   
        
    Note:
        min_std_multiplier and max_std_multiplier must be explicitly set since
        noise injection is essential for simulating MaskedFuzzySensor behavior.
    """
    
    def __init__(self, 
                 *args,
                 sensor_configs: list[list[str]], 
                 **kwargs):
        """
        Initialize the End2EndTransformer.
        
        Parameters
        ----------
        sensor_configs: list[list[str]]
            List of sensor configurations. Each inner list specifies which state
            features are observed (unmasked) in that sensor configuration.
            Features not in the list are masked as zero during training.
            
        *args:
            Additional positional arguments passed to MyTimesSeriesTransformer.
            
        **kwargs:
            Additional keyword arguments passed to MyTimesSeriesTransformer.
            Must include min_std_multiplier and max_std_multiplier.
            
        Raises
        ------
        ValueError
            If min_std_multiplier or max_std_multiplier is None, as these are
            essential for simulating MaskedFuzzySensor behavior.
        """
        # Disable Masking (this was originally used for MHLP)
        if 'masking_min_context' in args:
            print("Ignoring masking_min_context! Option not available for this End-2-End Estimator!")
        if 'masking_max_context' in args:
            print("Ignoring masking_max_context! Option not available for this End-2-End Estimator!")
        kwargs["masking_min_context"] = None
        kwargs["masking_max_context"] = None

        # Sanity check min_std_multiplier and max_std_multiplier
        if 'min_std_multiplier' in kwargs:
            if kwargs['min_std_multiplier'] is None:
                raise ValueError("min_std_multiplier must be set")
        if 'max_std_multiplier' in kwargs:
            if kwargs['max_std_multiplier'] is None:
                raise ValueError("max_std_multiplier must be set")

        super().__init__(*args, **kwargs)

        # Generate sensor masks
        self.state_vec_order_original = ["u", "v", "r", "x", "y", "psi", "p", "phi", "delta", "n"] 
        self.state_vec_order_encoded  = ["u", "v", "r", "x", "y", "p", "phi", "delta", "n", "psi_sin", "psi_cos"] 
        sensor_feature_masks = torch.zeros(len(sensor_configs), 11, dtype=bool, device=self.device)
        for sensor_idx, y_vec_order in enumerate(sensor_configs):
            sensor_feature_masks[sensor_idx] = self.generate_feature_mask(y_vec_order)
        self.register_buffer("sensor_feature_masks", sensor_feature_masks)        

    def generate_feature_mask(self, y_vec_order: list[str]) -> Tensor:
        """
        Convert a list of sensor features to a boolean mask for the state vector.
        
        This method creates a boolean mask that indicates which features in the
        encoded state vector are observed by the sensor. The mask is used during
        training to simulate partial observations.
        
        Parameters
        ----------
        y_vec_order: list[str]
            List of feature names that are observed by the sensor.
            Must be valid state vector features.
            
        Returns
        -------
        Tensor, shape (11,)
            Boolean mask where True indicates the feature is observed.
            The mask corresponds to the encoded state vector order.
            
        Raises
        ------
        AssertionError
            If any feature in y_vec_order is not in the state vector order.
        """
        feature_mask = torch.zeros(11, dtype=bool, device=self.device)
        for feature in y_vec_order:
            assert feature in self.state_vec_order_original, f"Feature {feature} not in state vector order!"
            if feature == "psi":
                feature_mask[self.state_vec_order_encoded.index("psi_sin")] = True
                feature_mask[self.state_vec_order_encoded.index("psi_cos")] = True
            else:
                feature_mask[self.state_vec_order_encoded.index(feature)] = True
        return feature_mask

    def prepare_input_target(self, batch: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        """
        Prepare input and target data for training.
        
        This method simulates the MaskedFuzzySensor behavior during training:
        1. Adds noise to the true state vector
        2. Randomly selects a sensor configuration
        3. Applies masking to simulate partial observations
        4. Prepares the input as [y, u] and target as encoded x
        
        Parameters
        ----------
        batch: tuple[Tensor, Tensor]
            Tuple containing (x, u) where:
            - x: Tensor, shape (batch_size, seq_len, n_states) - true states
            - u: Tensor, shape (batch_size, seq_len, n_controls) - control inputs
            
        Returns
        -------
        tuple[Tensor, Tensor]
            Tuple containing (A, B) where:
            - A: Tensor, shape (batch_size, seq_len, n_states + n_controls) - input [y, u]
            - B: Tensor, shape (batch_size, seq_len, n_states_encoded) - target encoded states
        """
        # Prepare input and target data
        x, u = batch
        b, l, d = x.shape # batch, seq-len, feature dim
        
        # Sensor logic (ie take measurements y=sensor(x)
        # 1. Add noise to the state vector from std in range [min_std_multiplier, max_std_multip
        y = x.clone()
        std_multiplier = torch.distributions.uniform.Uniform(self.min_std_multiplier, self.max_std_multiplier).sample((b,1,1))
        y += torch.randn_like(y) * self.noise_std_x.to(y) * std_multiplier.to(y.device)
        y = self.encode_x(y)
        
        # 2. Sample a sensor configuration for each sequence in batch and apply its masking 
        sensor_idxs = torch.randint(0, len(self.sensor_feature_masks), size=(x.shape[0],), device=self.device) 
        y = y[:,:,:] * self.sensor_feature_masks[sensor_idxs,:].unsqueeze(1).to(y.device) 
        
        # Prepare input control vector
        u += torch.randn_like(u) * self.noise_std_u.to(u)
        u = self.encode_u(u)
        A = torch.cat([y,u], dim=-1)
        
        # Target output
        x = self.encode_x(x)
        B = x
        return A,B

    def predict_with_nans(self, 
                         y: np.ndarray | Tensor, 
                         u: np.ndarray | Tensor, 
                         y_vec_order: list[str]) -> np.ndarray | Tensor:
        """
        Predict the full state sequence given measurements and control inputs.
        
        This method takes partial observations y and control inputs u, then predicts
        the full state sequence. The method handles the mapping from sensor space
        to state space and applies proper masking.
        
        Parameters
        ----------
        y: np.ndarray | Tensor, shape (seq_len, n_measurements)
            Measurement sequence. Can be numpy array or torch tensor.
            
        u: np.ndarray | Tensor, shape (seq_len, n_controls)
            Control input sequence. Can be numpy array or torch tensor.
            
        y_vec_order: list[str]
            List of feature names that are observed in the measurements.
            Must match the order of features in y.
            
        Returns
        -------
        np.ndarray | Tensor, shape (seq_len, n_states)
            Predicted full state sequence. Returns same type as input.
            
        Raises
        ------
        AssertionError
            If y or u don't have exactly 2 dimensions.
        ValueError
            If y is not a numpy array or torch tensor.
        AssertionError
            If any feature in y_vec_order is not in the state vector order.
        """
        # Validate input shapes
        assert len(y.shape) == 2, f"y must have 2 dimensions, got shape {y.shape}"
        assert len(u.shape) == 2, f"u must have 2 dimensions, got shape {u.shape}"

        # Handle input type conversion
        dtype_ = y.dtype
        if isinstance(y, np.ndarray):
            is_numpy = True
            y = torch.tensor(y, dtype=self.dtype_)
            u = torch.tensor(u, dtype=self.dtype_)
        elif isinstance(y, torch.Tensor):
            is_numpy = False
        else:
            raise ValueError(f"y must be numpy array or torch tensor, got {type(y)}")

        y = y.unsqueeze(0).to(self.device)
        u = u.unsqueeze(0).to(self.device)

        # Map sensor measurements to state space
        # Create a zero tensor for the full state and fill in observed features
        x_ = torch.zeros((1, y.shape[1], 10), dtype=y.dtype, device=y.device)
        for idx_y, feature_name in enumerate(y_vec_order):
            assert feature_name in self.state_vec_order_original, f"Feature {feature_name} not in state vector order!"
            idx_x = self.state_vec_order_original.index(feature_name)
            x_[:,:, idx_x] = y[:,:, idx_y]
        
        # Encode the state and control inputs
        x_ = self.encode_x(x_)
        u = self.encode_u(u)

        # Apply proper masking to simulate partial observations
        x_ *= self.generate_feature_mask(y_vec_order)
        A: Tensor = torch.cat([x_, u], dim=-1)
        
        # Forward pass through the model
        x_pred: Tensor = self.forward(A)
        assert x_pred.shape == x_.shape
        x_pred = self.decode_x(x_pred)[0, :, :]

        # Convert back to original type and format
        if is_numpy:
            x_pred = x_pred.detach().cpu().numpy()
        x_pred = x_pred.astype(dtype_)
        return x_pred

    def predict(self, 
               y: np.ndarray | Tensor, 
               u: np.ndarray | Tensor, 
               y_vec_order: list[str]) -> np.ndarray | Tensor:
        """
        Predict the current state given measurements and control inputs.
        
        This method predicts the current state from the measurement sequence.
        It calls predict_with_nans and returns only the last timestep.
        
        Parameters
        ----------
        y: np.ndarray | Tensor, shape (seq_len, n_measurements)
            Measurement sequence. Can be numpy array or torch tensor.
            
        u: np.ndarray | Tensor, shape (seq_len, n_controls)
            Control input sequence. Can be numpy array or torch tensor.
            
        y_vec_order: list[str]
            List of feature names that are observed in the measurements.
            Must match the order of features in y.
            
        Returns
        -------
        np.ndarray | Tensor, shape (n_states,)
            Predicted current state. Returns same type as input.
        """
        return self.predict_with_nans(y, u, y_vec_order)[..., -1, :]
    
    
class End2EndEstimator(EstimatorBase):
    """
    End-to-end state estimator using a trained transformer model.
    
    This estimator uses a trained End2EndTransformer to directly estimate
    the full state from partial sensor measurements. Unlike traditional
    Kalman filters, this approach learns the complete state estimation
    mapping from data during training

    """
    
    def __init__(self,
                 model: End2EndTransformer | str,
                 sensor: 'MaskedFuzzyStateSensor',
                 h: float,
                 max_context: int = 128,
                 integrator_method: str | None = None):
        """
        Initialize the End2EndEstimator.
        
        Parameters
        ----------
        model: End2EndTransformer | str
            Trained end-to-end transformer model. If a string is provided,
            it's treated as a checkpoint path and the model is loaded.
            
        sensor: MaskedFuzzyStateSensor
            Sensor model that provides the measurement function and
            projection methods. Must be a MaskedFuzzyStateSensor instance.
            
        h: float
            Time step size for simulation.
            
        max_context: int, default 128
            Maximum number of previous time steps to use as context
            for the transformer model.
            
        integrator_method: str | None, default None
            Integration method for position and heading angle correction.
            If None, no integration is applied.
            
        Raises
        ------
        ValueError
            If sensor is not a MaskedFuzzyStateSensor instance.
        """
        # Sanity checks
        if not isinstance(sensor, fmukf.simulation.sensors.MaskedFuzzyStateSensor):
            raise ValueError("sensor must be a MaskedFuzzyStateSensor")
        if isinstance(model, str):
            model = End2EndTransformer.load_from_checkpoint(model, strict=False)
        
        # Remember Parameters
        self.model: 'End2EndTransformer' = model
        self.sensor: 'MaskedFuzzyStateSensor' = sensor
        self.sensor_y_vec_order: list[str] = sensor.y_vec_order
        self.h: float = h
        self.max_context: int = max_context
        
        # Initialize history of measurements and inputs
        self.Y_history: list[np.ndarray] = []
        self.U_history: list[np.ndarray] = []
        
        # Initialize integrator method (if used)
        self.integrator_method: str | None = integrator_method
        if integrator_method is not None:
            self.xhat_prev: np.ndarray | None = None

    def __call__(self, 
                 y: np.ndarray, 
                 u: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        y: np.ndarray, shape (n_measurements,)
            Current measurement vector.
            
        u: np.ndarray, shape (n_controls,)
            Current control input vector.
            
        Returns
        -------
        np.ndarray, shape (n_states,)
            Estimated current state vector.
        """
        
        # Store the current measurements and inputs
        self.Y_history.append(y)
        self.U_history.append(u)
        Y = np.array(self.Y_history)
        U = np.array(self.U_history)
        
        # Clip the context to the max_context range (should be lower than what used in model training)
        if Y.shape[0] > self.max_context:
            Y = Y[-self.max_context:, :]
            U = U[-self.max_context:, :]

        # Project Y onto state space of X (any missing features will be masked as nan)
        Y_ = self.sensor.project_y_onto_x(Y)
        
        # Convert to tensors, add in batch dimension, and move to same device as model
        Y_ = torch.tensor(Y_, dtype=self.model.dtype_).unsqueeze(0).to(self.model.device)
        U_ = torch.tensor(U,  dtype=self.model.dtype_).unsqueeze(0).to(self.model.device)
        
        # Apply Feature Transformations to Y and U
        Y__ = self.model.encode_x(Y_)
        Y__ = torch.nan_to_num(Y__, nan=0.0)
        U__ = self.model.encode_u(U_)

        # Forward Pass
        A = torch.cat((Y__, U__), dim=-1)
        Bpred = self.model.forward(A)
        
        # Decode the predicted State estimate
        xhat = self.model.decode_x(Bpred)[0,-1,:]

        # If requested, correct the positions with integration
        if self.integrator_method is not None and self.xhat_prev is not None:
            xhat = self.correct_positions_with_integration(xhat, self.xhat_prev, self.integrator_method)
            self.xhat_prev = xhat
        
        # Convert to numpy array and return
        xhat = xhat.detach().cpu().numpy()
        return xhat