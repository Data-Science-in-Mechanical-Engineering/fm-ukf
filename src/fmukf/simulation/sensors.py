import numpy as np


class SensorModelBase:
    """Base class for sensor/observation models that implement the measurement equation y = g(x,u) + epsilon(x,u).
    
    This abstract base class defines the interface for sensor models used in state estimation.
    Sensor models transform the true state vector x and input vector u into noisy measurements y.
    The measurement equation follows the form: y = g(x,u) + epsilon(x,u), where:
    - g(x,u) is the deterministic measurement function (should be differentiable)
    - epsilon(x,u) is the measurement noise
    
    Child classes must implement the abstract methods g_function(), epsilon(), and getContiniousLinearCD().
    
    Main usage:
        >>> # Create a sensor instance
        >>> sensor = MySensor(EnvSim, random_seed=42)
        >>> 
        >>> # Get measurements at a single timestep
        >>> y = sensor(x, u, t=0.0)
        >>> 
        >>> # Get measurements without noise
        >>> y_clean = sensor(x, u, noise=False)
        >>> 
        >>> # Create linearized version around operating point
        >>> linear_sensor = sensor.MakeLinearInstance(x0, u0)
    """

    y_vec_order: list[str] # ["var1", "var2", ...] # Explicit Order of variables output vector
    y_space_definition: dict     # {"var1": {"units": "m/s", description: "The first variable..."}, "var2": ...} #<-- useful definitions for each system should go her
    
    name: str = None
    description: str = None

    def reset_rng(self, random_seed=None):
        """Reset the random number generator with a new seed.
        
        Args:
            random_seed: 
                New random seed for the RNG. If None, generates a random seed using secrets.
        """
        if random_seed is None:
            import secrets
            random_seed = secrets.randbits(32)
        self.rng = np.random.RandomState(random_seed)

    def __init__(self, random_seed):
        """Initialize the sensor model with a random seed.
        
        Args:
            random_seed: 
                Random seed for reproducible noise generation
        """
        self.reset_rng()
        # Explicit Order of variables output vector
        # y_space_definition: dict     # {"var1": {"units": "m/s", description: "The first variable..."}, "var2": ...} #<-- useful definitions for each system should go her

        # name: str = None
        # description: str = None
        # # random_seed = None # <-- Random Seed object to initialize the 

    def g_function(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Returns the deterministic measurement function y = g(x,u).
        
        This function should be deterministic and differentiable. It represents
        the ideal (noise-free) measurement given the true state and input.
        
        Args:
            x: 
                State vector, shape (d_states,)
            u: 
                Input vector, shape (d_inputs,)
                
        Returns:
            Measurement vector y = g(x,u), shape (d_measurements,)
            
        Raises:
            NotImplementedError: Must be implemented by child classes
        """
        raise NotImplementedError
        return y
        
    def epsilon(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Returns a sample of measurement noise epsilon(x,u).
        
        This function generates the noise component that gets added to the
        deterministic measurement g(x,u) to create the final noisy measurement.
        
        Args:
            x: 
                State vector, shape (d_states,)
            u: 
                Input vector, shape (d_inputs,)
                
        Returns:
            Noise vector epsilon(x,u), shape (d_measurements,)
            
        Raises:
            NotImplementedError: Must be implemented by child classes
        """
        raise NotImplementedError
        return epsilon
    
    def getContiniousLinearCD(self, x0: np.ndarray, u0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns the matrices C, D for continuous linearization around (x0, u0).
        
        The linearized measurement equation is: y = C*x + D*u
        
        Args:
            x0: 
                Operating point state vector, shape (d_states,)
            u0: 
                Operating point input vector, shape (d_inputs,)
                
        Returns:
            Tuple containing:
                - C: Measurement matrix, shape (d_measurements, d_states)
                - D: Input matrix, shape (d_measurements, d_inputs)
                
        Raises:
            NotImplementedError: Must be implemented by child classes
        """
        raise NotImplementedError
        return C, D
    
    def MakeLinearInstance(self, x0: np.ndarray, u0: np.ndarray) -> 'SensorModelBase':
        """Creates a linearized version of this sensor around the operating point (x0, u0).
        
        The returned sensor implements the linear measurement equation:
        y = C*(x - x0) + D*(u - u0) + y0
        
        Args:
            x0: 
                Operating point state vector, shape (d_states,)
            u0: 
                Operating point input vector, shape (d_inputs,)
                
        Returns:
            Linearized sensor instance with the same interface as the original
        """
        selfcopy = {key: self.__dict__[key] for key in self.__dict__}
        class LinearSensorClass(self.__class__):
            def __init__(self1):
                # Copy instance variables (ie the self from parent to child)
                for key in selfcopy:
                    self1.__dict__[key] = selfcopy[key]
                self1.x0 = x0
                self1.u0 = u0
                self1.y0 = self.g_function(x0, u0)
                self1.C, self1.D = self.getContiniousLinearCD(x0, u0)

            def g_function(self1, x, u):
                return self1.C@(x - self1.x0) + self1.D@(u - self1.u0) + self1.y0

        return LinearSensorClass()
    
    def plot_trajectory(self, t_arr: np.ndarray, y_arr: np.ndarray, label: str = "y"):
        """Plots a trajectory of sensor data.
        
        Args:
            t_arr: 
                Time array, shape (n_timesteps,)
            y_arr: 
                Sensor data array, shape (n_timesteps, d_measurements)
            label: 
                Label for the plot legend
                
        Raises:
            NotImplementedError: Must be implemented by child classes
        """
        raise NotImplementedError
    
    def __call__(self, x: np.ndarray, u: np.ndarray, t: float = None, noise: bool = True) -> np.ndarray:
        """Return a sample observation y = g(x,u) + epsilon(x,u).
        
        This is the main interface for getting measurements from the sensor.
        
        Args:
            x: 
                State vector, shape (d_states,)
            u: 
                Input vector, shape (d_inputs,)
            t: 
                Current time (optional, for time-dependent sensors)
            noise: 
                Whether to add noise to the measurement. If False, returns only g(x,u)
                
        Returns:
            Measurement vector y, shape (d_measurements,)
        """
        if noise:
            return self.g_function(x, u) + self.epsilon(x, u)
        else:
             return self.g_function(x, u)
        
    def project_y_onto_x(self, y: np.ndarray) -> np.ndarray | None:
        """Map sensor measurement vector y to state vector x.
        
        This function is useful when y contains some variables that are also part of x,
        and you want to plot them on the same graph. Variables that do not project
        onto x should be set to np.nan.
        
        Args:
            y: 
                Measurement vector, shape (d_measurements,) or (n_timesteps, d_measurements)
                
        Returns:
            State vector with projected measurements, shape (d_states,) or (n_timesteps, d_states),
            or None if there is no projection mapping
        """
        return None #<-- None means there is no mapping from y to x
        

class IdentitySensor(SensorModelBase):
    """Dummy sensor that returns y = x + 0 (no noise).
    
    This sensor simply returns the state vector without any noise or transformation.
    It's useful for testing and as a baseline for perfect measurements.
    
    Main usage:
        >>> # Create identity sensor for an environment
        >>> sensor = IdentitySensor(EnvSim)
        >>> 
        >>> # Get perfect measurements
        >>> y = sensor(x, u)  # y = x
    """
    
    def __init__(self, EnvSim: 'EnvSimBase', **kwargs):
        """Initialize the identity sensor.
        
        Args:
            EnvSim: 
                Environment simulator that defines the state space
            **kwargs: 
                Additional keyword arguments (ignored)
        """
        self.name = 'IdentitySensor'
        self.description = 'It just returns the original state vector'

        self.y_vec_order = EnvSim.state_vec_order
        self.y_state_definition = {var: {"units": EnvSim.state_space[var]["units"]} for var in EnvSim.state_space}
        super().__init__(random_seed=None)  # No random seed needed for identity sensor
        
    def epsilon(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Returns zero noise (perfect measurements).
        
        Args:
            x: 
                State vector, shape (d_states,)
            u: 
                Input vector, shape (d_inputs,)
                
        Returns:
            Zero noise vector, shape (d_states,)
        """
        return 0
    
    def g_function(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Returns the state vector directly (identity function).
        
        Args:
            x: 
                State vector, shape (d_states,)
            u: 
                Input vector, shape (d_inputs,)
                
        Returns:
            State vector x, shape (d_states,)
        """
        return x
    
    def project_y_onto_x(self, y: np.ndarray) -> np.ndarray:
        """Direct projection since y = x.
        
        Args:
            y: 
                Measurement vector (same as state vector), shape (d_states,) or (n_timesteps, d_states)
                
        Returns:
            State vector y, shape (d_states,) or (n_timesteps, d_states)
        """
        return y

    def getContiniousLinearCD(self, x0: np.ndarray, u0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns identity matrix C and zero matrix D.
        
        For identity sensor: y = x, so C = I and D = 0.
        
        Args:
            x0: 
                Operating point state vector, shape (d_states,)
            u0: 
                Operating point input vector, shape (d_inputs,)
                
        Returns:
            Tuple containing:
                - C: Identity matrix, shape (d_states, d_states)
                - D: Zero matrix, shape (d_states, d_inputs)
        """
        C = np.eye(len(x0))
        D = np.zeros((len(x0), len(u0)))  # Fixed: was (len(u0), len(u0))
        return C, D

class MaskedFuzzyStateSensor(SensorModelBase):
    """Sensor that observes a subset of state variables with additive Gaussian noise.
    
    This sensor implements partial state observation where only a subset of state
    variables are observed, and each observed variable has independent Gaussian noise
    with specified standard deviations.
    
    The measurement equation is: y = x[mask] + N(0, stds^2)
    where mask selects the observed state variables.
    
    Main usage:
        >>> # Define which states to observe and their noise levels
        >>> std_dict = {"x": 0.1, "y": 0.1, "psi": 1.0}  # Only observe position and heading
        >>> 
        >>> # Create sensor
        >>> sensor = MaskedFuzzyStateSensor(EnvSim, std_dict=std_dict)
        >>> 
        >>> # Get noisy measurements of subset of states
        >>> y = sensor(x, u)  # y contains only x, y, psi with noise
    """
    
    def __init__(self, EnvSim: 'EnvSimBase', std_dict: dict[str, float] = None, random_seed: int = None):
        """Initialize the masked fuzzy state sensor.
        
        Args:
            EnvSim: 
                Environment simulator that defines the state space
            std_dict: 
                Dictionary mapping state variable names to their noise standard deviations.
                Format: {"var1": std1, "var2": std2, ...}
                If None, defaults to no noise for all state variables.
            random_seed: 
                Random seed for noise generation
        """
        super().__init__(random_seed)
        self.reset_rng(random_seed)
        self.name = 'MaskedFuzzyStateSensor'
        self.description = f'Returns the variables with the following stds:\n{std_dict}'

        if std_dict is None:
            std_dict = {var:0 for var in EnvSim.state_vec_order}

        # self.plot_state_trajectory = EnvSim.plot_trajectory
        self.Envsim_state_vec_order = EnvSim.state_vec_order

        for var in std_dict:
            assert var in EnvSim.state_vec_order, f'"{var}" is not a known state vector variable. State variables in order: {EnvSim.state_vec_order}'
        
        self.y_vec_order =  [var for var in EnvSim.state_vec_order if var in std_dict] #<-- Force same order as original state_vector
        self.y_space_definition = {var: {"units": EnvSim.state_space[var]["units"],  "epsilon_std": std_dict[var]} for var in std_dict}

        self.index_mask = np.array([EnvSim.state_vec_order.index(var) for var in self.y_vec_order])
        self.stds = np.array([std_dict[var] for var in self.y_vec_order])

    def g_function(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Returns the subset of state variables that are observed.
        
        Args:
            x: 
                State vector, shape (d_states,)
            u: 
                Input vector, shape (d_inputs,)
                
        Returns:
            Observed state variables, shape (d_measurements,)
        """
        return x[self.index_mask]
    
    def epsilon(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Returns Gaussian noise with specified standard deviations.
        
        Args:
            x: 
                State vector, shape (d_states,)
            u: 
                Input vector, shape (d_inputs,)
                
        Returns:
            Gaussian noise vector, shape (d_measurements,)
        """
        return self.rng.normal(scale = self.stds)
    
    def getContiniousLinearCD(self, x0: np.ndarray, u0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns selection matrix C and zero matrix D.
        
        For masked sensor: y = x[mask], so C is a selection matrix and D = 0.
        
        Args:
            x0: 
                Operating point state vector, shape (d_states,)
            u0: 
                Operating point input vector, shape (d_inputs,)
                
        Returns:
            Tuple containing:
                - C: Selection matrix, shape (d_measurements, d_states)
                - D: Zero matrix, shape (d_measurements, d_inputs)
        """
        n = len(self.y_vec_order)
        C = np.zeros((n, len(x0)))
        D = np.zeros((n, len(u0)))
        for i,j in enumerate(self.index_mask):
            C[i,j] = 1
        return C, D

    def project_y_onto_x(self, y: np.ndarray) -> np.ndarray:
        """Maps measurement vector y back to full state vector x.
        
        Observed variables are placed in their correct positions in x,
        while unobserved variables are set to np.nan.
        
        Args:
            y: 
                Measurement vector, shape (d_measurements,) or (n_timesteps, d_measurements)
                
        Returns:
            State vector with projected measurements, shape (d_states,) or (n_timesteps, d_states)
        """
        projection_idxs = [self.Envsim_state_vec_order.index(var) for var in self.y_vec_order]
        num_vars_x = len(self.Envsim_state_vec_order)
        if len(y.shape) == 1: # case: y is just a vector (one time-step):
            x = np.zeros(num_vars_x)
            x[:] = np.nan
            x[projection_idxs] = y
        elif len(y.shape) == 2: # Case: y is a matrix (assume second dim is time-step)
            num_time_steps = y.shape[0]
            x = np.zeros((num_time_steps, num_vars_x))
            x[:,:] = np.nan
            x[:, projection_idxs] = y
        else:
            raise ValueError(f"Expected Input y to be vector (one time-step) or matrix (multiple time step). Got y with shape {y.shape}")
        return x


class FuzzyBridgeSensor(MaskedFuzzyStateSensor):
    """Sensor that simulates time-dependent noise levels, like going under bridges.
    
    This sensor extends MaskedFuzzyStateSensor with time-dependent noise levels.
    It alternates between two noise regimes:
    - Normal operation: lower noise levels
    - Bridge/occlusion events: higher noise levels
    
    The timing is controlled by t_no_bridge and t_bridge parameters.
    
    Main usage:
        >>> # Define noise levels for normal and bridge conditions
        >>> vars = {"x": (0.1, 20.0), "y": (0.0, 15.1)}  # (normal_std, bridge_std)
        >>> 
        >>> # Create sensor with timing
        >>> sensor = FuzzyBridgeSensor(EnvSim, vars, t_no_bridge=30, t_bridge=5)
        >>> 
        >>> # Use in simulation loop
        >>> for t in time_steps:
        ...     y = sensor(x, u, t)  # Noise level depends on current time
    """
    
    def __init__(self, EnvSim: 'EnvSimBase', vars: dict[str, tuple[float, float]], t_no_bridge: float, t_bridge: float, random_seed: int = None):
        """Initialize the fuzzy bridge sensor.
        
        Args:
            EnvSim: 
                Environment simulator that defines the state space
            vars: 
                Dictionary mapping state variable names to noise tuples.
                Format: {"var1": (std_normal, std_bridge), "var2": (std_normal, std_bridge), ...}
                Example: {"x": (0.1, 20.0), "y": (0.0, 15.1)}
            t_no_bridge: 
                Duration of normal operation phase in seconds
            t_bridge: 
                Duration of bridge/occlusion phase in seconds
            random_seed: 
                Random seed for noise generation
        """
        self.reset_rng(random_seed)
        """Basically the MaskedFuzzyStateSensor but stds that are time dependent, to simulate the vehicle repeatedly 
        'going under bridges' (in which the measurments for some variables become extra noisy). Specfically for 
        t_no_bridge seconds the MaskedFuzzyStateSensor uses the stds from the first index of the tuple, and 
        then for t_bridge seconds uses the stds from the second index the tuple.

        Format: vars={"var1": (std_no_bridge, std_bridge), "var2":...}
        e.g.{"x": (0.1, 20.0), "y": (0.0, 15.1)}
        """

        # Init regular MaskedFuzzyStateSensor with dummy stds
        super().__init__(EnvSim, {var:0 for var in vars}, random_seed=random_seed) 
        
        # Remember stds and timings 
        self.stds_no_bridge = np.array([vars[var][0] for var in self.y_vec_order])
        self.stds_bridge = np.array([vars[var][1] for var in self.y_vec_order])
        self.t_no_bridge = t_no_bridge
        self.t_bridge = t_bridge

        # Initialize state (no bridge)
        self.under_bridge = False
        self.stds = self.stds_no_bridge
        self.t_next = self.t_no_bridge # <-- The next time the event flips between bridge and no bridge

        # Set name and so on
        self.name = 'FuzzyBridgeSensor'
        self.description = f"""Every {t_no_bridge} s an occlusion/bridge event occurs for {t_bridge}s respectively,
        during which the stds of the following observed change from (std_no_bridge, std_bridge)accordinly:\n{vars}"""
        
    def __call__(self, x: np.ndarray, u: np.ndarray, t: float, noise: bool = True) -> np.ndarray:
        """Return a sample observation with time-dependent noise levels.
        
        This method overrides the parent __call__ to handle time-dependent
        noise switching between normal and bridge conditions.
        
        Args:
            x: 
                State vector, shape (d_states,)
            u: 
                Input vector, shape (d_inputs,)
            t: 
                Current time in seconds (required for timing the noise switches)
            noise: 
                Whether to add noise to the measurement
                
        Returns:
            Measurement vector y, shape (d_measurements,)
        """
        if t >= self.t_next:
            if self.under_bridge:  # Transition: Bridge --> No Bridge
                self.under_bridge = False
                self.t_next = t + self.t_no_bridge
                self.stds = self.stds_no_bridge
            else:  # Transition: No Bridge --> Bridge
                self.under_bridge = True  
                self.t_next = t + self.t_bridge
                self.stds = self.stds_bridge    
        return super().__call__(x, u, noise=noise)
