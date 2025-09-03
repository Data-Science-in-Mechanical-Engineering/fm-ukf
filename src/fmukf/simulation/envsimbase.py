import numpy as np
from scipy.integrate import solve_ivp
import scipy
from copy import copy 
from tqdm.auto import tqdm
from fmukf.simulation.controllers import ControllerBase
from fmukf.simulation.sensors import SensorModelBase

from IPython.display import display


class EnvSimBase:
    """Base class for simulation environments and dynamical systems.
    
    This class provides a framework for implementing dynamic system simulations
    with state estimation, sensor models, and controllers. Child classes must
    implement several abstract methods to define the specific dynamics and
    properties of their simulation environment.
    
    The simulation follows this loop:
        y_k    = sensor(x_k, u_(k-1)) 
        xhat_k = estimator(y_k, u_(k-1)) 
        u_k    = controller(xhat_k)
        x_(k+1) = sim_step(x_k, u_k, h)
    
    Example usage:
        # Create a simulation environment (child class)
        env = MySimulationEnv()
        
        # Define a simple constant controller
        def controller(t, xhat):
            return np.array([1.0, 0.0])  # constant input
        
        # Run simulation
        traj = env.simulate(
            x0=env.get_x(),              # initial state
            controller=controller,        # controller function
            h=0.1,                      # time step
            Nsteps=100                  # number of steps
        )
        
        # Access results
        times = traj["t"]               # time array
        states = traj["x"]              # state history
        inputs = traj["u"]              # input history
    
    Attributes to be defined by child classes:
        state_vec_order: List of state variable names in order they appear in state vector
        input_vec_order: List of input variable names in order they appear in input vector
        state_space: Dictionary defining properties of each state variable
        input_space: Dictionary defining properties of each input variable
        parameters: Dictionary of simulation parameters
        rng: Random number generator for sampling
        name: Name of the simulation environment
        description: Description of the simulation environment
    """
    
    ###############################################################################
    ######   Placeholder methods that need to be implemented by child classes  ####
    
    # Class attributes that must be defined by child classes
    state_vec_order: list[str] = None # ["var0", "var1", ...] #<-- Order of variables in state vector
    input_vec_order: list[str] = None # ["var0", "var1", ...] #<-- Order of variables in input vector
    
    # Properties of each variable in the state space vector go here. Required: "default_value" and "sample_range" (for random initial state sampling)
    state_space: dict = None    # {"var0": {"default_value": 0.0, "sample_range": (-2, +2), "bananas_range": (-10, 10), "units": "m/s", "description": "The first value"}, "var1":...}
    input_space: dict = None    # {"var0": {"default_value": 0.0, "sample_range": (-2, +2), "units": "m/s", "description": "The first value"}, "var1":...}

    def xdot(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Computes the state vector derivative dx/dt for a given state (x) and input (u) vector.
        
        This method defines the continuous-time dynamics of the system:
            xdot = f(x, u)
        
        Args:
            x: State vector, shape (d_states,)
            u: Input vector, shape (d_inputs,)
            
        Returns:
            State derivative vector dx/dt, shape (d_states,)
            
        Raises:
            NotImplementedError: Must be implemented by child classes
        """
        raise NotImplementedError("xdot method must be implemented by child classes")

    def getContinousLinearAB(self, x0: np.ndarray, u0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns the continuous linearization matrices A,B around (x0, u0) s.t.
            x_dot = A*x + B*u 
        
        Args:
            x0: Linearization point for state, shape (d_states,)
            u0: Linearization point for input, shape (d_inputs,)
            
        Returns:
            Tuple of (A, B) matrices where:
                - A is the state matrix, shape (d_states, d_states)
                - B is the input matrix, shape (d_states, d_inputs)
            
        Raises:
            NotImplementedError: Must be implemented by child classes
        """
        raise NotImplementedError("getContinousLinearAB method must be implemented by child classes")

    def getContinousLinearCD(self, x0: np.ndarray, u0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns the continuous linearization matrices C,D around (x0, u0) s.t.
            y = C*x + D*u 
        
        Args:
            x0: Linearization point for state, shape (d_states,)
            u0: Linearization point for input, shape (d_inputs,)
            
        Returns:
            Tuple of (C, D) matrices where:
                - C is the output matrix, shape (n_outputs, d_states)
                - D is the feedthrough matrix, shape (n_outputs, d_inputs)
            
        Raises:
            NotImplementedError: Must be implemented by child classes
        """
        raise NotImplementedError("getContinousLinearCD method must be implemented by child classes")

    def visualize_trajectory(self, traj_dict: dict[str, np.ndarray], sensor=None, plot_estimate=True, hv_backend="bokeh"):
        """Visualize trajectory data with true states, estimates, and sensor measurements.
        
        This is an abstract method that should be implemented by child classes to provide
        appropriate visualization for the specific simulation environment.
        
        Args:
            traj_dict: Dictionary containing trajectory data with keys:
                - "t": time array, shape (Nsteps,)
                - "x": true state array, shape (Nsteps, d_states)
                - "u": input array, shape (Nsteps, d_inputs)
                - "y": measurement array, shape (Nsteps, n_measurements)
                - "xhat": state estimate array, shape (Nsteps, d_states)
            sensor: Sensor model object that can project measurements to state space
                   (optional, default: None)
            plot_estimate: Whether to include state estimates in visualization
                          (optional, default: True)
            hv_backend: Holoviews backend to use for visualization
                       (optional, default: "bokeh")
        
        Returns:
            holoviews.Layout: A holoviews layout object containing the visualization plots
            
        Raises:
            NotImplementedError: Must be implemented by child classes
        """
        raise NotImplementedError("visualize_trajectory method must be implemented by child classes")  

    def getInitialInput(self, x0: np.ndarray) -> np.ndarray:
        """Function to initialize the first controller input vector u_(-1) to feed to sensor and estimator model during simulation.

        The reason why this is necessary is because of the order of the simulation loop:
            y_k    = sensor(x_k, u_(k-1)) 
            xhat_k = estimator(y_k, u_(k-1)) 
            u_k    = controller(xhat_k)

        For case k=0, the sensor and estimator need u_(-1). This method tells how to initialize it.
        
        Args:
            x0: Initial state vector, shape (d_states,)
            
        Returns:
            Initial input vector u_(-1), shape (d_inputs,)
            
        Raises:
            NotImplementedError: Must be implemented by child classes
        """
        raise NotImplementedError("getInitialInput method must be implemented by child classes")

    @classmethod
    def get_default_parameters(cls) -> dict:
        """Returns default parameters of the simulation model that define its transition dynamics.
        
        Returns:
            Dictionary of default parameters
            
        Raises:
            NotImplementedError: Must be implemented by child classes
        """
        raise NotImplementedError("get_default_parameters method must be implemented by child classes")
        
    ###############################################################################

    def __init__(self,
                 parameters: dict = None,
                 random_seed: int = None):
        """Initialize the simulation environment.
        
        Args:
            parameters: Dictionary of simulation parameters. If None, uses default parameters
                       from get_default_parameters()
            random_seed: Seed for random number generator. If None, generates a random seed
        """
        # Initialize Parameters
        if parameters is None:
            parameters = self.get_default_parameters()
        self.parameters = parameters 
        
        # Reset Random Number Generator
        self.reset_rng(random_seed)

        # Check if necessary properties have been implemented by child
        if self.state_vec_order is None: 
            raise NotImplementedError('"self.state_vec_order" has not been defined in child')
        if self.input_vec_order is None: 
            raise NotImplementedError('"self.input_vec_order" has not been defined in child')
        if self.state_space is None: 
            raise NotImplementedError('"self.state_space" has not been defined in child')
        if self.input_space is None: 
            raise NotImplementedError('"self.input_space" has not been defined in child')

        # Validate state space configuration
        for var in self.state_vec_order:
            assert "default_value" in self.state_space[var], f'Missing "default_value" for variable "{var}" in self.state_space'
            assert "sample_range" in self.state_space[var],  f'Missing "sample_range" for variable "{var}" in self.state_space'
            assert "bananas_range" in self.state_space[var], f'Missing "bananas_range" for variable "{var}" in self.state_space'
            
        # Validate input space configuration
        for var in self.input_vec_order:
            assert "default_value" in self.input_space[var], f'Missing "default_value" for variable "{var}" in self.input_space'
            assert "sample_range" in self.input_space[var],  f'Missing "sample_range" for variable "{var}" in self.input_space'

        # Check if name and description are there
        if "name" not in self.__dict__: 
            self.name = "MISSING NAME"
        if "description" not in self.__dict__: 
            self.description = "MISSING DESCRIPTION"
        

    ###############################################################################

    def simulate(self,
                 x0: np.ndarray,
                 controller: ControllerBase = None,
                 u0: np.ndarray = None,
                 state_estimator: 'EstimatorBase' = None,
                 sensor: 'SensorModelBase' = None,
                 h: float = 1.0,
                 Nsteps: int = 500,
                 silent: bool = False, 
                 visualize: bool = True, 
                 tqdm_kwargs: dict = None) -> dict[str, np.ndarray]:
        """Simulates system for h*Nsteps seconds from initial state x0.
        
        The simulation follows this loop:
            t_k     = t_(k-1) + h
            u_k     = controller(xhat_(k-1))     
            x_k     = sim_step(x_(k-1), u_(k-1))
            y_k     = sensor(x_k, u_k)
            xhat_k  = estimator(y_k, u_k)
        
        Args:
            x0 (np.ndarray): 
                Initial state vector, shape (d_states,). 
                Use .get_x() or .sample_x() to generate appropriate initial states.
            controller (callable or np.ndarray): 
                Either a callable function u = controller(t, xhat) that returns input vector 
                shape (d_inputs,), or a constant input vector shape (d_inputs,) applied at 
                every timestep.
            u0 (np.ndarray, optional): 
                Initial action vector, shape (d_inputs,). If None (default), uses 
                self.getInitialInput(x0) for callable controllers or the controller value 
                itself for constant controllers.
            state_estimator (callable, optional): 
                Function xhat = state_estimator(y, u) that estimates states.
                If None, assumes perfect state estimation (xhat = x).
            sensor (callable, optional): 
                Function y = sensor(x, u, t) returning measurement vector.
                If None, assumes perfect measurements (y = x).
            h (float): 
                Time step length in seconds (default: 1.0)
            Nsteps (int): 
                Number of simulation steps (default: 500)
            silent (bool): 
                If True, suppresses progress bars and non-critical output (default: False)
            visualize (bool): 
                Whether to display trajectory plots after simulation (default: True)
            tqdm_kwargs (dict, optional): 
                Additional arguments for tqdm progress bar
        
        Returns:
            Dictionary with keys "t", "u", "x", "y", "xhat" containing numpy arrays:
                - "t": time array, shape (Nsteps,)
                - "u": input array, shape (Nsteps, d_inputs)
                - "x": state array, shape (Nsteps, d_states)
                - "y": measurement array, shape (Nsteps, n_measurements)
                - "xhat": state estimate array, shape (Nsteps, d_states)
        """
        if tqdm_kwargs is None:
            tqdm_kwargs = {}

        if silent:
            tqdm_kwargs["disable"] = True # <-- Stops a tqdm progress bar from appearing
            visualize = False
  
        # Format Estimator
        if (state_estimator is None) and (not silent):
                print("No State_Estimator was specified: Enforcing xhat_t = x_t (ie perfect state estimate)")
        
        # Format Sensor
        if (sensor is None) and (not silent):
            print("sensor = None: Will force y_t = x_t (ie perfect state observations)")

        # Format initial Controller Input 
        if u0 is None:
            if not callable(controller):
                u0 = controller # Case where controller is just a constant input so we just use that there
            else:
                u0 = self.getInitialInput(x0)
                if not silent:
                    print("No initial input was given: Using default u0 = self.getInitialInput(x0)")

        trajectory = [] # Trajectory tuple
        t = 0
        u = copy(u0)
        x = copy(x0)
        for _ in tqdm(range(Nsteps), desc="Simulating Timesteps", **tqdm_kwargs):
            y = sensor(x, u, t) if sensor is not None else x
            xhat = state_estimator(y, u) if state_estimator is not None else x
           
            # Check if state vector has gone 'Bananas' (ie extreme values, indicating simulation has gone out of whack)
            self.is_bananas(x, raise_exception=True)
            trajectory.append((t, u, x, y, xhat))

            # Next timestep
            t += h
            u = controller(t, xhat) if callable(controller) else controller 
            x = self.sim_step(x, u, h)

        # Convert trajectory tuples to dictionary format
        traj_dict = self.traj_tuples_2_dict(trajectory)
        
        if visualize:
            if state_estimator is None:
                plot_estimate = False
            else:
                plot_estimate = True
            display(self.visualize_trajectory(traj_dict, sensor, plot_estimate=plot_estimate))

        return traj_dict

    def sim_step(self, x: np.ndarray, u: np.ndarray, h: float) -> np.ndarray:
        """Simulates system for h seconds with initial state x under constant action/input u.
        
        Uses scipy.integrate.solve_ivp with RK45 method to integrate the continuous dynamics.
        
        Args:
            x: Initial state vector, shape (d_states,)
            u: Constant input vector, shape (d_inputs,)
            h: Time step length
            
        Returns:
            Final state vector after h seconds, shape (d_states,)
        """
        x, u = copy(x), copy(u) # Should hopefully brute enforce "pass by value"
        solution = solve_ivp(lambda t,y: self.xdot(y, u), t_span=(0, h), y0=x, method='RK45')
        return solution.y[:,-1]
  

    def is_bananas(self, x: np.ndarray, raise_exception: bool = True) -> bool:
        """Checks if variable values of state vector x are outside the defined limits of 'bananas_range' 
        in self.state_space indicating the simulation has become numerically unstable.
        
        Args:
            x: State vector to check, shape (d_states,)
            raise_exception: If True, raises an exception when state goes bananas. 
                           If False, just returns True/False
            
        Returns:
            True if state has gone bananas, False otherwise
            
        Raises:
            AssertionError: If raise_exception=True and state has gone bananas
        """
        for idx, var in enumerate(self.state_vec_order):
            min_val, max_val = self.state_space[var]["bananas_range"]
            if x[idx] < min_val or max_val < x[idx]:
                if raise_exception:
                    assert False, f"Variable {var}={x[idx]} has gone Bananas! Should be inside self.state_space['{var}']['bananas_range'] = {self.state_space[var]['bananas_range']}"
                return True
        return False

    def traj_tuples_2_dict(self, traj_tuples: list[tuple]) -> dict[str, np.ndarray]:
        """Converts a trajectory in form of list of tuples to a dictionary of arrays.
        
        Args:
            traj_tuples: List of tuples [(t0, u0, x0, y0, xhat0), (t1, u1, x1, y1, xhat1), ...]
            
        Returns:
            Dictionary with keys "t", "u", "x", "y", "xhat" containing numpy arrays:
                - "t": time array, shape (Nsteps,)
                - "u": input array, shape (Nsteps, d_inputs)
                - "x": state array, shape (Nsteps, d_states)
                - "y": measurement array, shape (Nsteps, n_measurements)
                - "xhat": state estimate array, shape (Nsteps, d_states)
        """
        traj_dict = {}
        traj_dict["t"]     = np.array([t    for (t, u, x, y, xhat) in traj_tuples])
        traj_dict["u"]     = np.array([u    for (t, u, x, y, xhat) in traj_tuples])
        traj_dict["x"]     = np.array([x    for (t, u, x, y, xhat) in traj_tuples])
        traj_dict["y"]     = np.array([y    for (t, u, x, y, xhat) in traj_tuples])
        traj_dict["xhat"]  = np.array([xhat for (t, u, x, y, xhat) in traj_tuples])
        return traj_dict


    def reset_rng(self, random_seed: int = None):
        """Resets the random number generator for sampling new states with appropriate seed.
        
        If random_seed=None then will generate new 'truly' random seed that is safe to use in a thread.
        
        Args:
            random_seed: Seed for random number generator. If None, generates a random seed
        """
        if random_seed is None:
            import secrets
            random_seed = secrets.randbits(32)
        self.rng = np.random.RandomState(random_seed)


    def get_x(self, **kwargs) -> np.ndarray:
        """Returns a state vector x with default values specified in self.state_space, 
        except for any variable specified as a keyword argument.
        
        Examples: 
            .get_x(), 
            get_x(u=1, v=10), 
            get_x(**{"u":1, "v":1})
            
        Args:
            **kwargs: Keyword arguments specifying values for specific state variables
            
        Returns:
            State vector with specified values, shape (d_states,)
            
        Raises:
            AssertionError: If any keyword argument is not a valid state variable
        """
        for arg in kwargs: # Check if variables are valid
            assert arg in self.state_vec_order, f'"{arg}" is not a valid state variable. State variables in order: {self.state_vec_order} (see self.state_space for details)'

        state = {var: self.state_space[var]["default_value"] for var in self.state_vec_order}
        state |= kwargs # Overwrite default values specified by keyword arguments
        return np.array([state[var] for var in self.state_vec_order], dtype=float)

    def get_u(self, **kwargs) -> np.ndarray:
        """Returns an input vector u with default values specified in self.input_space, 
        except for any variable specified as a keyword argument.
        
        Examples: 
            .get_u(), 
            get_u(delta=0.1), 
            get_u(**{"delta":0.1, "n":1})
            
        Args:
            **kwargs: Keyword arguments specifying values for specific input variables
            
        Returns:
            Input vector with specified values, shape (d_inputs,)
            
        Raises:
            AssertionError: If any keyword argument is not a valid input variable
        """
        for arg in kwargs: # Check if variables are valid
            assert arg in self.input_vec_order, f'"{arg}" is not a valid input variable. Input variables in order: {self.input_vec_order} (see self.input_space for details).'

        input_dict = {var: self.input_space[var]["default_value"] for var in self.input_vec_order}
        input_dict |= kwargs # Overwrite default values specified by keyword arguments
        return np.array([input_dict[var] for var in self.input_vec_order], dtype=float)
    
    def sample_x(self) -> np.ndarray:
        """Returns a random state vector with values uniformly sampled in range defined by 
        "sample_range" for each state variable in self.state_space.
        
        Returns:
            Random state vector, shape (d_states,)
        """
        state_var_dict = {}
        for var in self.state_vec_order:
            min_val, max_val = self.state_space[var]['sample_range']
            state_var_dict[var] = self.rng.uniform(low=min_val, high=max_val)
        return self.get_x(**state_var_dict)

    def sample_u(self) -> np.ndarray:
        """Returns a random input vector with values uniformly sampled in range defined by 
        "sample_range" for each input variable in self.input_space.
        
        Returns:
            Random input vector, shape (d_inputs,)
        """
        input_var_dict = {}
        for var in self.input_vec_order:
            min_val, max_val = self.input_space[var]['sample_range']
            input_var_dict[var] = self.rng.uniform(low=min_val, high=max_val)
        return self.get_u(**input_var_dict)
    

    ###############################################################################
    #########   Methods for creating a linear simulation instance  ################
    

    def getDiscreteLinearABCD(self, x0: np.ndarray, u0: np.ndarray, h: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return matrices A,B,C,D for the discrete linearization around (x0, u0) with time-step h s.t.
            x_k+1 = A_d*x_k + B_d*u_k 
            y_k   = C_d*x_k + D_d*u_k
        
        Args:
            x0: Linearization point for state, shape (d_states,)
            u0: Linearization point for input, shape (d_inputs,)
            h: Time step for discretization
            
        Returns:
            Tuple of (A_d, B_d, C_d, D_d) discrete-time matrices:
                - A_d: discrete state matrix, shape (d_states, d_states)
                - B_d: discrete input matrix, shape (d_states, d_inputs)
                - C_d: discrete output matrix, shape (n_outputs, d_states)
                - D_d: discrete feedthrough matrix, shape (n_outputs, d_inputs)
        """
        A, B = self.getContinousLinearAB(x0, u0)
        C, D = self.getContinousLinearCD(x0, u0)
        A_d, B_d, C_d, D_d, _ = scipy.signal.cont2discrete((A, B, C, D), h)
        return A_d, B_d, C_d, D_d

    def MakeLinearInstance(self, x0: np.ndarray, u0: np.ndarray):
        """Return a new class with linear dynamics w.r.t. x0, u0, and h.
        
        Creates a linearized version of the simulation around the specified operating point.
        
        Args:
            x0: Linearization point for state, shape (d_states,)
            u0: Linearization point for input, shape (d_inputs,)
            
        Returns:
            LinearEnvSim instance with linearized dynamics
        """
        selfcopy = {key: self.__dict__[key] for key in self.__dict__}
        class LinearEnvSim(self.__class__):
            def __init__(self1):
                # Copy state and all instance variables
                for key in selfcopy:
                    self1.__dict__[key] = selfcopy[key]

                # Remember linearization point
                self1.x0, self1.u0 = x0, u0
                
                # Continuous Time Matrices
                self1.A, self1.B = self.getContinousLinearAB(x0, u0)

                # Init Cache for discrete Time Matrices (we will use these later)
                self1.A_d, self1.B_d, self1.h = None, None, 0

                # Override default values to x0 and u0
                for idx, var in enumerate(self.state_vec_order):
                    self1.state_space[var]["default_value"] = x0[idx]
                for idx, var in enumerate(self.input_vec_order):
                    self1.input_space[var]["default_value"] = u0[idx]

            def sim_step(self1, x, u, h):
                """Returns x_(k+1) = A@(x_k - x0) + B@(u_k - u0) + x0 for timestep length h"""
                if h != self1.h:
                    # Different timestep length, need to recompute the DISCRETE version
                    self1.A_d, self1.B_d, _, _, self1.h = scipy.signal.cont2discrete((self1.A, self1.B, 0, 0), h)
                return self1.A_d@(x - self1.x0) + self1.B_d@(u - self1.u0) + self1.x0
            
        return LinearEnvSim()






# def initialize(object):
#     # Checks if object is a class type or function and initializes if need be
#     if inspect.isclass(object) or inspect.isfunction(object):
#         return object()
#     else: # just a regular object, instance of a class (including call method)
#         return object

# def allocate_idxs(rank, Nworkers, Nruns):
#     # Calculate the base number of runs per worker
#     base_count = Nruns // Nworkers
#     # Calculate the remainder (the extra runs to be distributed)
#     remainder = Nruns % Nworkers
    
#     # Determine the starting index for this worker
#     if rank < remainder:
#         start = rank * (base_count + 1)
#         end = start + base_count + 1
#     else:
#         start = remainder * (base_count + 1) + (rank - remainder) * base_count
#         end = start + base_count
    
#     # Return the list of indexes for this worker
#     return list(range(start, end))







# import numpy as np
# from fmukf.utils import save_to_file, load_from_file
# from tqdm.auto import tqdm
# import inspect
# from fmukf.utils import run_with_timeout, cloudpickle_dumps64
# import h5py
# import os
# from fmukf.utils import runWorkersInParrallel, FileProgressBar


# def generateTrajectoryData(EnvSim: 'EnvSimBase',
#                            controller,         # Either constant vector u, or calleable u=controller(t, x)
#                            estimator = None,   # Calleable function xhat_(k+1) = estimator(y_k)
#                            sensor = None,      # Calleable function: y_k = sensor(x_k, u_k)
#                            Nruns  = 10,        # Total number of trajectories
#                            Nsteps = 5000,      # Number of time steps 
#                            h      = 0.25,      # time length of each time_step
#                            x0     = None,      # Initial state, or function to sample initial_start state
#                            filename = "TrajectoryData.h5", 
#                            num_workers = 4,    # Number of parrallel processes to use
#                            dtype = np.float32, # The dtype with which stored in h5 (use None for no conversion)
#                             ) -> list[dict[str, np.ndarray]]:
#     """Simulate multiple trajectories and save as file. Returns trajectory (list of dictionary of numpy arrays)
    
#     Basically just repeats EnvSim.Simulate(x0, controller, Nsteps, h, ...) so see documentation for that

#     TODO: DOCUMENTATION
#     """
    
#     # Input Formatting
#     if x0 is None:
#         print("No function or value specified to set x0! Will use random initial value instead")

#     # Remember parameters (and the actual objects) with cloudpickle
#     params = dict(Nruns=Nruns, Nsteps=Nsteps, h=h)
#     objects = dict(EnvSim=EnvSim, controller=controller, estimator=estimator, sensor=sensor, x0=x0)
#     with h5py.File(filename, "w") as db: #<-- Note this will delet any file with that name
#         db.attrs["base64cloudpicke"] = cloudpickle_dumps64({"params": params, "objects": objects})

#     # Launch a ProgressBar
#     pbar_file = filename.split("h5")[0]+"__progressbar.txt"
#     pbar = FileProgressBar(filename=pbar_file, total=Nruns)

#     # Worker process that simulates multiple trajectories (is to be launched in parrallel)
#     def TrajectoryWorker(rank):
#         # Create temporary h5 file for each worker (we will merge them later)
#         filename_ = filename.split(".h5")[0] + f"__temp__{rank}.h5"
#         with h5py.File(filename_, "w") as db:
#             db.attrs["rank"] = rank

#         # Run several trajectories (each with index as indicated here)
#         for traj_idx in allocate_idxs(rank, num_workers, Nruns):

#             # Attemp each simulation for max 10 times (otherwise abort)
#             attempts = 0
#             while attempts <= 10: 
#                 try:
#                     #Actually simulate the data (with initialized objects)
#                     EnvSim_ = initialize(EnvSim)
#                     traj_tuples = EnvSim_.simulate( h=h, Nsteps=Nsteps, silent=True,
#                         x0              = initialize(x0) if x0 is not None else EnvSim_.sample_x(),
#                         controller      = initialize(controller),
#                         state_estimator = initialize(estimator),
#                         sensor          = initialize(sensor)
#                     )
#                     traj_dict = EnvSim_.traj_tuples_2_dict(traj_tuples)
                    
#                     # Save results to h5 file
#                     traj_dict = EnvSim_.traj_tuples_2_dict(traj_tuples)
#                     with h5py.File(filename_, "r+") as db:
#                         for key, value in traj_dict.items():
#                             if dtype is not None: # Convert to certain datatype if specified
#                                 value = value.astype(dtype)
#                             db.create_dataset(f"traj_{traj_idx}/{key}", data=value)

#                     # Signal progress bar that run is complete (by appending idx to file)
#                     with open(pbar_file, "a+") as f:
#                         f.write(f"{traj_idx}\n")
#                     break
#                 except Exception as e:
#                     print(f"Trajectory #{traj_idx} failed for reason below (trying again):\n{e}")
#                     attempts += 1
#             else:
#                 raise RuntimeError("Failed to run the same trajectory 10 times in a row! Something must realbe broken?")

#     # Run these trajectory workers in parrallel (meanwhile is just a function that is to be repeatedly called)
#     results = runWorkersInParrallel(TrajectoryWorker, num_workers, meanwhile=pbar.check_progress)

#     print("Finished computing. Now merging temporary h5 files into one.")
#     # Merge temporary h5 files into one
#     with h5py.File(filename, "r+") as db:
#         for rank in range(num_workers):
#             tmp_filename = filename.split(".h5")[0] + f"__temp__{rank}.h5"
#             with h5py.File(tmp_filename, "r") as db_temp:
#                 for key in db_temp:
#                     db_temp.copy(key, db)
    
#     # (try to) delete temporary files
#     print("Finished merging. Now deleting temporary files.")
#     try:
#         os.remove(pbar_file)
#     except:
#         pass
#     for rank in range(num_workers):
#         try:
#             tmp_filename = filename.split(".h5")[0] + f"__temp__{rank}.h5"
#             os.remove(tmp_filename)
#         except FileNotFoundError:
#             pass
#     print("Done :)")

