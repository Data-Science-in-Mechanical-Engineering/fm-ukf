from fmukf.simulation.envsimbase import EnvSimBase
import numpy as np
from copy import copy
import holoviews as hv
from jax import jacfwd
import jax.numpy as jnp
from copy import deepcopy
import holoviews as hv



class Container(EnvSimBase):
    """
    Simulates a Container ship based on Son og Nomoto (1982). On the Coupled Motion of Steering and  Rolling of a High Speed Container Ship, Naval Architect of Ocean Engineering, 20: 73-83. From J.S.N.A. , Japan, Vol. 150, 1981.
    The implementation is adapted from Fossen et al's Matlab implentation as part of the Marine System Simulator (MSS) library
    (see: https://github.com/cybergalactic/MSS/blob/master/VESSELS/models/container.m for implementation details).
    
    This class implements a high-fidelity 6-DOF (degree of freedom) container ship model with coupled roll and steering dynamics.
    It serves as the foundation for testing various estimation algorithms (UKF, FMUKF) in the codebase. The model includes
    detailed hydrodynamic forces, rudder dynamics, and propeller thrust modeling based on empirical coefficients.
    
    The state vector consists of 10 variables:
        - u: Surge velocity (forward motion) [m/s]
        - v: Sway velocity (sideways motion) [m/s]
        - r: Yaw rate (turning rate) [rad/s]
        - x: North position [m]
        - y: East position [m]
        - psi: Yaw angle (heading) [rad]
        - p: Roll rate [rad/s]
        - phi: Roll angle [rad]
        - delta: Rudder angle [rad]
        - n: Propeller shaft velocity [rpm]
    
    The input vector consists of 2 variables:
        - delta_c: Commanded rudder angle [rad]
        - n_c: Commanded propeller shaft velocity [rpm]
    
    Example usage:
        # Example usage:
        # Create container ship model with default parameters
        >>> ship = Container()
        
        # Get random initial state
        >>> x0 = ship.sample_x()
        
        # Define inputs (rudder angle and shaft velocity)
        >>> u = np.array([np.radians(5.0), 80.0])  # 5 degrees rudder, 80 rpm
        
        # Simulate one step
        >>> h = 0.1  # time step in seconds
        >>> x1 = ship.sim_step(x0, u, h)
        
        # Run a full simulation with a controller
        >>> ctrlr = lambda t, x: np.array([np.sin(20*t), 80.0])
        >>> traj = ship.simulate(x0, controller=ctrlr, h=1.0, Nsteps=1000)
    
    Notes:
        - Throughout the codebase we use modified Container with a scaled state space vector (see .ScaledContainer)
    """
    
    def __init__(self, parameters:dict = None, random_seed:int = None) -> None:
        """Initialize the Container ship simulation environment.
        
        Args:
            parameters (dict, optional):
                Dictionary containing ship parameters. If None, default parameters are used.
                Default: None
            random_seed (int, optional):
                Random seed for reproducibility. Default: None
        
        Notes:
            - Sets up state space with 10 state variables: [u, v, r, x, y, psi, p, phi, delta, n]
            - Sets up input space with 2 control inputs: [delta, n]
            - Defines ranges for sampling, absolute limits, and "bananas" detection
        """

        # Environment Name and Description
        self.name = "Container"
        self.description = "Original Container"

        # Initialize Parameters
        if parameters is None:
            parameters = self.get_default_parameters()
        else:
            parameters = self.initialize_parameters(parameters)

        # Statevec definition
        self.state_vec_order = ["u", "v", "r", "x", "y", "psi", "p", "phi", "delta", "n"] 
        rot_sample_range = (-np.radians(1), np.radians(1))  # Maximum initial rad/s range for yaw and roll rate
        rot_bananas_range = (-2*np.pi, + 2*np.pi)
        # absolute_range defines the range, outside the physics and simulators break down
        # bananas_range defines the limits outside which a value is deemed "bananas" ie the simulation has numerically gone whack and the run will
        self.state_space = {
            "u": {
                "default_value": 0.1,
                "units": "m/s",
                "description": "Surge Velocity",
                "absolute_range": (0.01, np.inf),
                "sample_range": (0.01, 10),
                "bananas_range": (0, 40)
            },
            "v": {
                "default_value": 0.0,
                "units": "m/s",
                "description": "Sway Velocity",
                "sample_range": (0.0, 1),
                "bananas_range": (-20, 20)
            },
            "r": {
                "default_value": 0.0,
                "units": "rad/s",
                "description": "Yaw Rate",
                "sample_range": rot_sample_range,
                "bananas_range": rot_bananas_range
            },
            "x": {
                "default_value": 0.0,
                "units": "m",
                "description": "North Position",
                "sample_range": (0, 0),
                "bananas_range": (-10**6, +10**6)
            },
            "y": {
                "default_value": 0.0,
                "units": "m",
                "description": "East Position",
                "sample_range": (0, 0),
                "bananas_range": (-10**6, +10**6)
            },
            "psi": {
                "default_value": 0.0,
                "units": "rad",
                "description": "Yaw Angle",
                "sample_range": (-np.pi, +np.pi),
                "bananas_range": (-30 * np.pi, +30 * np.pi)
            },  # Note: Earth reference frame?
            "p": {
                "default_value": 0.0,
                "units": "rad/s",
                "description": "Roll Rate",
                "sample_range": rot_sample_range,
                "bananas_range": rot_bananas_range
            },
            "phi": {
                "default_value": 0.0,
                "units": "rad",
                "description": "Roll Angle",
                "sample_range": (-np.pi / 4, np.pi / 4),
                "bananas_range": (-np.pi / 2, +np.pi / 2)
            },
            "delta": {
                "default_value": 0.0,
                "units": "rad",
                "description": "Rudder Angle",
                "sample_range": (
                    -np.radians(parameters["delta_max"]),
                    np.radians(parameters["delta_max"])
                ),
                "bananas_range": (
                    -np.radians(parameters["delta_max"] + 2),
                    np.radians(parameters["delta_max"] + 2)
                )
            },
            "n": {
                "default_value": 1.0,
                "units": "",
                "description": "Shaft velocity",
                "absolute_range": (0.01, parameters["n_max"]),
                "sample_range": (0.1, parameters["n_max"]),
                "bananas_range": (0, parameters["n_max"] + 1)
            },
        }
        
        # Order and definitions of state vector variables
        self.input_vec_order = ["delta", "n"]
        self.input_space = {
            "delta": {
                "default_value": 0.0,
                "units": "rad",
                "description": "Desired Rudder Angle",
                "absolute_range": (-np.radians(parameters["delta_max"]), np.radians(parameters["delta_max"])
                ),
                "sample_range": (-np.radians(parameters["delta_max"]), np.radians(parameters["delta_max"])
                )
            }, 
            "n": {
                "default_value": 1.0,
                "units": "",
                "description": "Propeller shaft velocity",
                "absolute_range": (0.1, parameters["n_max"]),
                "sample_range": (0.1, parameters["n_max"])
            }
        }

        super().__init__(parameters=parameters, random_seed=random_seed)

    @classmethod
    def get_default_parameters(cls) -> dict:
        """Returns default parameters of the container model that define its transition dynamics as a dictionary.
        
        Returns:
            dict:
                Dictionary containing all ship parameters including hydrodynamic coefficients,
                mass properties, and geometric parameters
        
        Notes:
            - Some remaining parameters like intertia tensors need to be computed in initialize_parameters()
            - Parameters were copied from https://github.com/cybergalactic/MSS/blob/master/VESSELS/models/shipModels/dataContainer.m

        """
        # Default parameters of the container model that define its transition dynamics as a dictionary.
        # Some remaining parameters like intertia tensors need to be computed in initialize_parameters()

        default_container_parameters = {
        "L": 175,
        "delta_max": 10.0,
        "Ddelta_max": 5.0,
        "n_max": 160.0,
        "m": 0.00792,
        "mx": 0.000238,
        "my": 0.007049,
        "Ix": 0.0000176,
        "alphay": 0.05,
        "lx": 0.0313,
        "ly": 0.0313,
        "Iz": 0.000456,
        "Jx": 0.0000034 * 10, # Artifically made the roll inertia bigger to slow down the roll dynamics
        "Jz": 0.000419,
        "g": 9.81,
        "nabla": 21222,
        "AR": 33.0376,
        "Delta": 1.8219,
        "D": 6.533,
        "rho": 1025.0,
        "t": 0.175,
        "T": 0.0005,
        "Xuu": -0.0004226,
        "Xvr": -0.00311,
        "Xrr": 0.00020,
        "Xphiphi": -0.00020,
        "Xvv": -0.00386,
        "Kv": 0.0003026,
        "Kr": -0.000063,
        "Kp": -0.0000075,
        "Kphi": -0.000021,
        "Kvvv": 0.002843,
        "Krrr": -0.0000462,
        "Kvvr": -0.000588,
        "Kvrr": 0.0010565,
        "Kvvphi": -0.0012012,
        "Kvphiphi": -0.0000793,
        "Krrphi": -0.000243,
        "Krphiphi": 0.00003569,
        "Yv": -0.0116,
        "Yr": 0.00242,
        "Yp": 0.0,
        "Yphi": -0.000063,
        "Yvvv": -0.109,
        "Yrrr": 0.00177,
        "Yvvr": 0.0214,
        "Yvrr": -0.0405,
        "Yvvphi": 0.04605,
        "Yvphiphi": 0.00304,
        "Yrrphi": 0.009325,
        "Yrphiphi": -0.001368,
        "Nv": -0.0038545,
        "Nr": -0.00222,
        "Np": 0.000213,
        "Nphi": -0.0001424,
        "Nvvv": 0.001492,
        "Nrrr": -0.00229,
        "Nvvr": -0.0424,
        "Nvrr": 0.00156,
        "Nvvphi": -0.019058,
        "Nvphiphi": -0.0053766,
        "Nrrphi": -0.0038592,
        "Nrphiphi": 0.0024195,
        "kk": 0.631,
        "epsilon": 0.921,
        "xR": -0.5,
        "wp": 0.184,
        "tau": 1.09,
        "xp": -0.526,
        "cpv": 0.0,
        "cpr": 0.0,
        "ga": 0.088,
        "cRr": -0.156,
        "cRrrr": -0.275,
        "cRrrv": 1.96,
        "cRX": 0.71,
        "aH": 0.237,
        "zR": 0.033,
        "xH": -0.48
        }

        default_container_parameters = cls.initialize_parameters(default_container_parameters)
        return deepcopy(default_container_parameters)
    
    # Define parameters within their own abstract class first and save then

    @classmethod
    def initialize_parameters(self, parameters:dict) -> dict:
        """Initializes the parameter object by setting the independent parameters and computing the dependent parameters.
        
        Args:
            parameters (dict):
                Dictionary containing ship parameters
        
        Returns:
            dict:
                Updated parameters dictionary with computed dependent parameters
        
        Notes:
            - Computes GM (metacentric height) from ship length
            - Computes mass matrix elements from basic mass and inertia properties
        """
        # Dependent parameters
        parameters["GM"] = 0.3 / parameters["L"]

        # Masses and moments of inertia
        parameters["m11"] = (parameters["m"] + parameters["mx"])
        parameters["m22"] = (parameters["m"] + parameters["my"])
        parameters["m32"] = -parameters["my"] * parameters["ly"]
        parameters["m42"] = parameters["my"] * parameters["alphay"]
        parameters["m33"] = (parameters["Ix"] + parameters["Jx"])
        parameters["m44"] = (parameters["Iz"] + parameters["Jz"])
        return parameters

    def xdot(self, x: np.ndarray, u: np.ndarray, npp=np) -> np.ndarray:  
        """Returns time derivative vector for the container ship dynamics.
        
        This method implements the core differential equations that govern the container ship's motion.
        It computes the time derivatives of all state variables based on the current state and inputs.
        The equations include complex hydrodynamic effects, rudder forces, propeller thrust, and coupling
        between different degrees of freedom (especially roll-yaw coupling).
        
        The `npp` parameter allows switching between standard NumPy and JAX's NumPy implementation.
        This dual implementation approach is crucial for our codebase because:
        1. Standard NumPy is faster for regular simulation runs
        2. JAX's NumPy enables automatic differentiation for linearization and gradient-based methods
        3. The same core dynamics can be used for both simulation and analytical derivatives
        
        Args:
            x (np.ndarray, shape: [10]):
                State vector [u, v, r, x, y, psi, p, phi, delta, n]
            u (np.ndarray, shape: [2]):
                Input vector [delta_c, n_c] (commanded rudder angle and shaft velocity)
            npp (module, optional):
                Numpy implementation to use. Use jax.numpy for analytical differentiation.
                Default: np
        
        Returns:
            np.ndarray, shape: [10]:
                Time derivatives of state variables
        
        Notes:
            - When using JAX (npp=jnp), the function becomes differentiable, enabling automatic
              computation of Jacobians for linearization and control design
        
        Example:
            # Using with standard NumPy for simulation
            x_state = np.array([...])  # Current state
            u_input = np.array([0.1, 80])  # Rudder angle and shaft speed
            dx_dt = ship.xdot(x_state, u_input)  # Get derivatives
            
            # Using with JAX for linearization
            from jax import numpy as jnp
            dx_dt_jax = ship.xdot(x_state, u_input, npp=jnp)  # Differentiable version
        """
        
        # Make copies for var naAming conflicts and to enforce "pass by value" (we )
        x, ui = copy(x), copy(u)

        if len(x) != 10:
            raise ValueError('x-vector must have dimension 10!')
        if len(ui) != 2:
            raise ValueError('u-vector must have dimension 2!')

        U = npp.sqrt(x[0]**2 + x[1]**2)  # surface speed (m/s)
        W = self.parameters["rho"] * self.parameters["g"] * self.parameters["nabla"] / (self.parameters["rho"] * self.parameters["L"]**2 * U**2 / 2)

        assert U > 0
        assert x[9] > 0
        # Non-dimensional states and inputs
        delta_c = ui[0]
        n_c = ui[1] / 60 * self.parameters["L"] / U
        u = x[0] / U
        v = x[1] / U
        p = x[6] * self.parameters["L"] / U
        r = x[2] * self.parameters["L"] / U
        phi = x[7]
        psi = x[5]
        delta = x[8]
        n = x[9] / 60 * self.parameters["L"] / U

        # Rudder saturation and dynamics
        if abs(npp.squeeze(delta_c)) >= self.parameters["delta_max"] * npp.pi / 180:
            delta_c = npp.sign(delta_c) * self.parameters["delta_max"] * npp.pi / 180

        delta_dot = delta_c - delta
        if abs(delta_dot) >= self.parameters["Ddelta_max"] * npp.pi / 180:
            delta_dot = npp.sign(delta_dot) * self.parameters["Ddelta_max"] * npp.pi / 180

        # Shaft velocity saturation and dynamics
        n_c = n_c * U / self.parameters["L"]
        n = n * U / self.parameters["L"]
        if abs(n_c) >= self.parameters["n_max"] / 60:
            n_c = npp.sign(n_c) * self.parameters["n_max"] / 60

        if n > 0.3:
            Tm = 5.65 / n
        else:
            Tm = 18.83

        n_dot = 1 / Tm * (n_c - n) * 60

        # Clip max rate of change 
        self.parameters["ndot_max"] = 20
        if abs(n_dot) > self.parameters["ndot_max"]:
            n_dot = npp.sign(n_dot) * self.parameters["ndot_max"]

        # Calculation of state derivatives
        vR = self.parameters["ga"] * v + self.parameters["cRr"] * r + self.parameters["cRrrr"] * r**3 + self.parameters["cRrrv"] * r**2 * v
        uP = u * ((1 - self.parameters["wp"]) + self.parameters["tau"] * ((v + self.parameters["xp"] * r)**2 + self.parameters["cpv"] * v + self.parameters["cpr"] * r))
        J = uP * U / (n * self.parameters["D"])
        KT = 0.527 - 0.455 * J
        uR = uP * self.parameters["epsilon"] * npp.sqrt(1 + 8 * self.parameters["kk"] * KT / (npp.pi * J**2))
        alphaR = delta + npp.arctan(vR / uR)
        FN = -((6.13 * self.parameters["Delta"]) / (self.parameters["Delta"] + 2.25)) * (self.parameters["AR"] / self.parameters["L"]**2) * (uR**2 + vR**2) * npp.sin(alphaR)
        T = 2 * self.parameters["rho"] * self.parameters["D"]**4 / (U**2 * self.parameters["L"]**2 * self.parameters["rho"]) * KT * n * npp.abs(n)

        # Forces and moments
        X = self.parameters["Xuu"] * u**2 + (1 - self.parameters["t"]) * T + self.parameters["Xvr"] * v * r + self.parameters["Xvv"] * v**2 + self.parameters["Xrr"] * r**2 + self.parameters["Xphiphi"] * phi**2 + self.parameters["cRX"] * FN * npp.sin(delta) + (self.parameters["m"] + self.parameters["my"]) * v * r

        Y = self.parameters["Yv"] * v + self.parameters["Yr"] * r + self.parameters["Yp"] * p + self.parameters["Yphi"] * phi + self.parameters["Yvvv"] * v**3 + self.parameters["Yrrr"] * r**3 + self.parameters["Yvvr"] * v**2 * r + self.parameters["Yvrr"] * v * r**2 + self.parameters["Yvvphi"] * v**2 * phi + self.parameters["Yvphiphi"] * v * phi**2 + self.parameters["Yrrphi"] * r**2 * phi + self.parameters["Yrphiphi"] * r * phi**2 + (1 + self.parameters["aH"]) * FN * npp.cos(delta) - (self.parameters["m"] + self.parameters["mx"]) * u * r

        K = self.parameters["Kv"] * v + self.parameters["Kr"] * r + self.parameters["Kp"] * p + self.parameters["Kphi"] * phi + self.parameters["Kvvv"] * v**3 + self.parameters["Krrr"] * r**3 + self.parameters["Kvvr"] * v**2 * r + self.parameters["Kvrr"] * v * r**2 + self.parameters["Kvvphi"] * v**2 * phi + self.parameters["Kvphiphi"] * v * phi**2 + self.parameters["Krrphi"] * r**2 * phi + self.parameters["Krphiphi"] * r * phi**2 - (1 + self.parameters["aH"]) * self.parameters["zR"] * FN * npp.cos(delta) + self.parameters["mx"] * self.parameters["lx"] * u * r - W * self.parameters["GM"] * phi

        N = self.parameters["Nv"] * v + self.parameters["Nr"] * r + self.parameters["Np"] * p + self.parameters["Nphi"] * phi + self.parameters["Nvvv"] * v**3 + self.parameters["Nrrr"] * r**3 + self.parameters["Nvvr"] * v**2 * r + self.parameters["Nvrr"] * v * r**2 + self.parameters["Nvvphi"] * v**2 * phi + self.parameters["Nvphiphi"] * v * phi**2 + self.parameters["Nrrphi"] * r**2 * phi + self.parameters["Nrphiphi"] * r * phi**2 + (self.parameters["xR"] + self.parameters["aH"] * self.parameters["xH"]) * FN * npp.cos(delta)

        # Dimensional state derivatives xdot = [ u v r x y psi p phi delta n ]'
        detM = self.parameters["m22"] * self.parameters["m33"] * self.parameters["m44"] - self.parameters["m32"]**2 * self.parameters["m44"] - self.parameters["m42"]**2 * self.parameters["m33"]

        xdot = npp.array([X * (U**2 / self.parameters["L"]) / self.parameters["m11"],
                -((-self.parameters["m33"] * self.parameters["m44"] * Y + self.parameters["m32"] * self.parameters["m44"] * K + self.parameters["m42"] * self.parameters["m33"] * N) / detM) * (U**2 / self.parameters["L"]),
                ((-self.parameters["m42"] * self.parameters["m33"] * Y + self.parameters["m32"] * self.parameters["m42"] * K + N * self.parameters["m22"] * self.parameters["m33"] - N * self.parameters["m32"]**2) / detM) * (U**2 / self.parameters["L"]**2),
                (npp.cos(psi) * u - npp.sin(psi) * npp.cos(phi) * v) * U,
                (npp.sin(psi) * u + npp.cos(psi) * npp.cos(phi) * v) * U,
                npp.cos(phi) * r * (U / self.parameters["L"]),
                ((-self.parameters["m32"] * self.parameters["m44"] * Y + self.parameters["m32"] * self.parameters["m42"] * N + K * self.parameters["m22"] * self.parameters["m44"] - K * self.parameters["m42"]**2) / detM) * (U**2 / self.parameters["L"]**2),
                p * (U / self.parameters["L"]),
                delta_dot,
                n_dot])

        return xdot

    def getInitialInput(self, x0: np.ndarray) -> np.ndarray:
        """Extract initial input values from state vector.
        
        Args:
            x0 (np.ndarray, shape: [10]):
                Initial state vector [u, v, r, x, y, psi, p, phi, delta, n]
        
        Returns:
            np.ndarray, shape: [2]:
                Initial input vector [delta, n] extracted from state vector
        
        Notes:
            - Since state vector contains true state of all controllable elements
              with same variable name, we just need to read that off there
        """
        # Since state vector contains true state of all controllable elements
        # with same variable name, we just need to read that off there
        #return self.get_u(**{var:x0[self.state_vec_order.index[var]] for var in self.input_vec_order})
        return np.array([x0[self.state_vec_order.index(var)] for var in self.input_vec_order])

    def getContinousLinearAB(self, x0: np.ndarray, u0: np.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Compute continuous-time linear state space matrices A and B.
        
        Args:
            x0 (np.ndarray, shape: [10]):
                Linearization point for state vector
            u0 (np.ndarray, shape: [2]):
                Linearization point for input vector
        
        Returns:
            tuple:
                A (jnp.ndarray, shape: [10, 10]): State matrix (∂ẋ/∂x)
                B (jnp.ndarray, shape: [10, 2]): Input matrix (∂ẋ/∂u)
        
        Notes:
            - Uses JAX for automatic differentiation to compute Jacobians
            - Returns matrices for continuous-time linear system: x_dot = A*x + B*u
            - The matrices represent partial derivatives of the state derivatives with respect to
              state and input variables
            - This method demonstrates the power of JAX's automatic differentiation - computing
              these Jacobians manually would be extremely tedious and error-prone
        
        Example:
            # Linearize around straight-line motion at 5 m/s
            >>> x_eq = np.array([5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 80.0])
            >>> u_eq = np.array([0.0, 80.0])  # Zero rudder, constant propeller
            
            # Get linear model
            >>> A, B = ship.getContinousLinearAB(x_eq, u_eq)
            
            # Now A and B can be used for linear control design or Kalman filtering
        """
        # Load jax.numpy arrays 
        x0, u0 = jnp.array(x0), jnp.array(u0)
        
        # Get xdot function with jax.numpy instead of numpy to allow for analytic differentiation
        xdot = lambda x, u : self.xdot(x, u, npp=jnp)

        # To complete a time-continues linear state space formulation 
        # x_dot = A*x + b*u 
        # y = C*x + d*u

        A = jacfwd(xdot, 0)(x0, u0)  # this is the derivative of x_dot w.r.t. all states x
        B = jacfwd(xdot, 1)(x0, u0)  # this is the derivative of x_dot w.r.t. all states x
        return A, B


    def visualize_trajectory(self,
                            traj_dict: dict,
                            sensor=None,
                            plot_estimate: bool=True,
                            wrap_psi: bool=True,
                            hv_backend: str="bokeh"):

        """Visualize trajectory data with true states, estimates, and sensor measurements.
        
        Creates a comprehensive visualization including:
        - Position plot (North vs East)
        - Individual state variable plots over time
        - Input plots (rudder angle and shaft velocity)
        - Sensor measurements (if available)
        - State estimates (if plot_estimate=True)
        
        Args:
            traj_dict (dict[str, np.ndarray]):
                Dictionary containing trajectory data with keys:
                - "t": time array (np.ndarray, shape: [n_timesteps])
                - "x": true state array (np.ndarray, shape: [n_timesteps, 10])
                - "u": input array (np.ndarray, shape: [n_timesteps, 2])
                - "y": measurement array (np.ndarray, shape: [n_timesteps, n_measurements])
                - "xhat": state estimate array (np.ndarray, shape: [n_timesteps, 10])
            sensor (object, optional):
                Sensor model object that can project measurements to state space.
                This object must implement project_y_onto_x() method to map sensor
                measurements to the state space for visualization.
                Default: None
            plot_estimate (bool, optional):
                Whether to include state estimates in visualization. Set to False
                if no estimates are available or to simplify the plots.
                Default: True
            wrap_psi (bool, optional):
                Whether to wrap psi (yaw angle) to [0, 360] degrees. This improves
                readability when the ship performs multiple turns.
                Default: True
            hv_backend (str, optional):
                HoloViews backend to use for plotting. Options include "bokeh" (interactive),
                "matplotlib" (static), and "plotly" (interactive with different features).
                Default: "bokeh"
        
        Returns:
            holoviews.Layout:
                A holoviews layout object containing the visualization plots.
                This can be displayed in Jupyter notebooks or exported to HTML.
                
        """
        # import pandas as pd
        hv.extension(hv_backend)
        # Prepare trajectory data
        trajsDicts = {}
        
        # True state trajectory
        traj_x = deepcopy(traj_dict)
        if wrap_psi:
            traj_x["x"][:, self.state_vec_order.index("psi")] = traj_x["x"][:, self.state_vec_order.index("psi")] % 360
        trajsDicts["state"] = traj_x
        
        # Sensor measurements (if available)
        if sensor is not None and (sensor.project_y_onto_x(traj_dict["y"][0,:]) is not None):
            if wrap_psi:
                if "psi" in sensor.y_vec_order:
                    traj_dict["y"][:, sensor.y_vec_order.index("psi")] = traj_dict["y"][:, sensor.y_vec_order.index("psi")] % 360
            
            traj_y = deepcopy(traj_dict)
            traj_y["x"] = sensor.project_y_onto_x(traj_dict["y"])
            traj_y["kind"] = "scatter"
            trajsDicts["sensor"] = traj_y
        
        # State estimates (if requested)
        if plot_estimate:
            traj_xhat = deepcopy(traj_dict)
            traj_xhat["x"] = traj_xhat["xhat"]
            trajsDicts["estimate"] = traj_xhat
        
        # Create plots
        subplotDict = {}
        
        # Position plot (North vs East)
        x_x = traj_x["x"][:, self.state_vec_order.index("x")]
        x_y = traj_x["x"][:, self.state_vec_order.index("y")]
        subplotDict["North vs East"] = hv.Curve((x_y, x_x), label="state").opts(
            title="North x vs East y",
            ylabel=f"x / {self.state_space['x']['units']}",
            xlabel=f"y / {self.state_space['y']['units']}",
            show_grid=True, data_aspect=1.0, width=400, height=300
        )
        
        # Add sensor data to position plot if available
        if "sensor" in trajsDicts:
            y_x = trajsDicts["sensor"]["x"][:, self.state_vec_order.index("x")]
            y_y = trajsDicts["sensor"]["x"][:, self.state_vec_order.index("y")]
            subplotDict["North vs East"] *= hv.Scatter((y_y, y_x), label="sensor").opts(marker="o", size=3)
        
        # Add estimate data to position plot if available
        if "estimate" in trajsDicts:
            xhat_x = trajsDicts["estimate"]["x"][:, self.state_vec_order.index("x")]
            xhat_y = trajsDicts["estimate"]["x"][:, self.state_vec_order.index("y")]
            subplotDict["North vs East"] *= hv.Curve((xhat_y, xhat_x), label="estimate")
        
        # Individual state variable plots
        for idx, var in enumerate(self.state_vec_order):
            subplotDict[var] = hv.Curve((traj_x["t"], traj_x["x"][:, idx]), label="state").opts(
                title=f"{self.state_space[var]['description']} - {var}",
                xlabel="t / s",
                ylabel=f"{var} / {self.state_space[var]['units']}",
                width=400, height=300
            )
            
            # Add sensor data if available
            if "sensor" in trajsDicts:
                subplotDict[var] *= hv.Scatter((trajsDicts["sensor"]["t"], trajsDicts["sensor"]["x"][:, idx]), 
                                              label="sensor").opts(marker="o", size=3)
            
            # Add estimate data if available
            if "estimate" in trajsDicts:
                subplotDict[var] *= hv.Curve((trajsDicts["estimate"]["t"], trajsDicts["estimate"]["x"][:, idx]), 
                                            label="estimate")
        
        # Input plots (only for the first trajectory to avoid clutter)
        for idx, var in enumerate(self.input_vec_order):
            subplotDict[var] = hv.Curve((traj_x["t"], traj_x["u"][:, idx]), label="input").opts(
                title=f"{self.input_space[var]['description']} - {var}",
                xlabel="t / s",
                ylabel=f"{var} / {self.input_space[var]['units']}",
                width=400, height=300
            )
        
        return hv.Layout([subplotDict[key] for key in subplotDict]).opts(shared_axes=False).cols(3)
       



class ScaledContainer(Container):
    """Wrapped Container class where the features of the state space have been modified into more readable ranges.
    
    This class provides a scaled version of the Container ship model where state variables are transformed
    to have more intuitive units and ranges. For example, positions are represented in kilometers instead of
    meters, and angles in degrees instead of radians. This scaling significantly makes plotting and comparing
    values more intuitiv e(e.g. degrees instead of radians).
    
    The scaling is transparent to the user - the dynamics remain identical to the original Container model,
    but all inputs and outputs are automatically scaled to the more convenient units. This is achieved through
    the scale_x/unscale_x and scale_u/unscale_u methods that convert between original and scaled representations.
    
    Example usage:
        # Create a scaled container ship
        >>> ship = ScaledContainer()
        
        # Initial state and control input (in scaled units)
        >>> x0 = ship.sample_x()  # Position in km, angles in degrees
        >>> u = np.array([5.0, 80.0])  # Rudder angle in degrees, shaft speed in rpm
        
        # Simulate (all calculations happen in original units internally)
        >>> x1 = ship.sim_step(x0, u, h=1.0)
        
        # Simulate a controller
        >>> ctrlr = lambda t, x: np.array([np.sin(20*t), 80.0])
        >>> traj = ship.simulate(x0, controller=ctrlr, h=1.0, Nsteps=1000)
       
    
    Notes:
        - This is the preferred class to use throughout the benchmark for improved
        - The scaling parameters are defined in the __init__ method as self.scale_params
    """


    def __init__(self, *args, **kwargs):
        """Initialize the ScaledContainer with scaled state and input spaces.
        
        Args:
            *args:
                Arguments passed to parent Container class
            **kwargs:
                Keyword arguments passed to parent Container class
        
        Notes:
            - Scales state variables to more readable ranges (e.g., km instead of meters)
            - Updates state and input space definitions with new units and ranges
            - Maintains the same dynamics as the original Container but with scaled coordinates
        """
        super().__init__(*args, **kwargs)

        # Name and description
        self.name = "ScaledContainer"
        self.description = "Original Container with scaled environmentspace"
        
        # Compute mean and scaling vectors
        deg = np.radians(1)
        
        # var: (mean, scale, unit) s.t. val --> (val - mean)/scale with "units"
        self.scale_params = {"state": {"u":(0,1,"m/s"), "v":(0,1,"m/s"), "r":(0,deg,"deg"),
                                       "x":(0,1000,"km"), "y":(0,1000,"km"), "psi":(0,deg,"deg"),
                                       "p":(0,deg,"deg"), "phi":(0,deg,"deg"), "delta":(0,deg,"deg"), "n":(0,1,"rpm")}, # "n":(0,160,"power fraction")
                             "input": {"delta":(0,deg,"deg"), "n":(0,1,"rpm")}}  # "n":(0,160,"power fraction")
        self.x_mean   = np.array([self.scale_params["state"][var][0] for var in self.state_vec_order])
        self.x_scales = np.array([self.scale_params["state"][var][1] for var in self.state_vec_order])
        self.u_mean   = np.array([self.scale_params["input"][var][0] for var in self.input_vec_order])
        self.u_scales = np.array([self.scale_params["input"][var][1] for var in self.input_vec_order])
        
        # Scale Fields in State Space definition
        self.old_state_space = copy(self.state_space)
        self.state_space     = copy(self.state_space)
        for var in self.scale_params["state"]:
            for key, value in list(self.state_space[var].items()):
                if key in set(["default_value", "absolute_range", "sample_range", "bananas_range"]):
                    
                    # Function to scale a specific index for that value
                    scalefunc = lambda val : (val - self.scale_params["state"][var][0]) / self.scale_params["state"][var][1] if not np.isinf(val) else val
                    if type(value) == tuple:
                        value = tuple(scalefunc(val) for val in value)
                    else:
                        value = scalefunc(value)
                    self.state_space[var][key] = value
                    
            # Overwrite Units
            self.state_space[var]["units"] = self.scale_params["state"][var][2]
        
        self.old_input_space = copy(self.input_space)
        self.input_space     = copy(self.input_space)
        for var in self.scale_params["input"]:
            for key, value in list(self.input_space[var].items()):
                if key in set(["default_value", "absolute_range", "sample_range", "bananas_range"]):
                    
                    # Function to scale a specific index for that value
                    scalefunc = lambda val : (val - self.scale_params["input"][var][0]) / self.scale_params["input"][var][1] if not np.isinf(val) else val
                    if type(value) == tuple:
                        value = tuple(scalefunc(val) for val in value)
                    else:
                        value = scalefunc(value)
                    self.input_space[var][key] = value

    def scale_x(self, x: np.ndarray) -> np.ndarray:
        """Scale state vector from original units to scaled units.
        
        Args:
            x (np.ndarray, shape: [10]):
                State vector in original units
        
        Returns:
            np.ndarray, shape: [10]:
                State vector in scaled units
        """
        return (x - self.x_mean) / self.x_scales

    def unscale_x(self, x: np.ndarray) -> np.ndarray:
        """Unscale state vector from scaled units back to original units.
        
        Args:
            x (np.ndarray, shape: [10]):
                State vector in scaled units
        
        Returns:
            np.ndarray, shape: [10]:
                State vector in original units
        """
        return x * self.x_scales + self.x_mean
    
    def scale_u(self, u: np.ndarray) -> np.ndarray:
        """Scale input vector from original units to scaled units.
        
        Args:
            u (np.ndarray, shape: [2]):
                Input vector in original units
        
        Returns:
            np.ndarray, shape: [2]:
                Input vector in scaled units
        """
        return (u - self.u_mean) / self.u_scales

    def unscale_u(self, u: np.ndarray) -> np.ndarray:
        """Unscale input vector from scaled units back to original units.
        
        Args:
            u (np.ndarray, shape: [2]):
                Input vector in scaled units
        
        Returns:
            np.ndarray, shape: [2]:
                Input vector in original units
        """
        return u * self.u_scales + self.u_mean
    
    def xdot(self, x: np.ndarray, u: np.ndarray, npp=np) -> np.ndarray:            
        """Returns time derivative vector for the scaled container ship dynamics.
        
        Args:
            x (np.ndarray, shape: [10]):
                State vector in scaled units [u, v, r, x, y, psi, p, phi, delta, n]
            u (np.ndarray, shape: [2]):
                Input vector in scaled units [delta_c, n_c]
            npp (module, optional):
                Numpy implementation to use.
                Default: np
        
        Returns:
            np.ndarray, shape: [10]:
                Time derivatives of state variables in scaled units
        
        Notes:
            - Unscales inputs and states, computes dynamics using parent class,
              then rescales the derivatives
        """
        x = self.unscale_x(x)
        u = self.unscale_u(u)

        xdot = super().xdot(x, u, npp=npp)
        xdot = xdot / npp.array(self.x_scales)
        return xdot
    
class ConstantVelocityContainer(ScaledContainer):
    """ScaledContainer involving a constant velocity forward model (To be only used with Constant Velocity Kalman Filters)
       Here  the transition function of the original Container class is simply replaced with a constant velocity forward model.
    """
    
    def sim_step(self, x: np.ndarray, u: np.ndarray, h: float) -> np.ndarray:
        """Simulate one step using constant velocity model.
        
        This method implements a simplified kinematic model for the container ship where velocities
        remain constant and positions are updated through basic integration. Unlike the full Container
        model which uses complex nonlinear differential equations, this method uses simple algebraic
        updates based on the current velocities.
        
        The key equations implemented are:
        - North position update: x += h * (u*cos(psi) - v*sin(psi))
        - East position update: y += h * (u*sin(psi) + v*cos(psi))
        - Heading update: psi += h * r
        - Roll angle update: phi += h * p
        
        All other state variables (velocities, rates, and inputs) remain unchanged.
        
        Args:
            x (np.ndarray, shape: [10]):
                Current state vector in scaled units [u, v, r, x, y, psi, p, phi, delta, n]
            u (np.ndarray, shape: [2]):
                Input vector in scaled units [delta_c, n_c] (ignored in this model)
            h (float):
                Time step size in seconds
        
        Returns:
            np.ndarray, shape: [10]:
                Next state vector in scaled units
        
        Notes:
            - Uses simplified constant velocity dynamics instead of full ship dynamics
            - Assumes constant surge and sway velocities (no acceleration)
            - Updates position based on velocity components transformed to world coordinates
            - Updates angles based on angular rates (simple integration)
            - Keeps control inputs constant (no actuator dynamics)
            - The division by 1000 in position updates accounts for the km scaling
        
        Example:
            # Initialize state with forward velocity and turning rate
            x = np.zeros(10)
            x[0] = 5.0  # 5 m/s forward
            x[2] = 0.1  # Small turning rate
            
            # Step forward in time (input is ignored)
            x_next = simple_ship.sim_step(x, np.zeros(2), h=0.1)
            
            # x_next will show updated position and heading
        """

        # Make into dictionary for sanity
        x_ = {var: x[idx] for idx, var in enumerate(self.state_vec_order)}

        # Speeds in m/s
        V_north =  x_['u'] * np.cos(np.radians(x_['psi']))  -  x_['v'] * np.sin(np.radians(x_['psi']))
        V_east  =  x_['u'] * np.sin(np.radians(x_['psi']))  +  x_['v'] * np.cos(np.radians(x_['psi']))

        # Next time-step
        x__ = {'u': x_['u'],
            'v':  x_['v'],
            'r':  x_['r'],
            'x':  x_['x'] + h * V_north / 1000,
            'y':  x_['y'] + h * V_east / 1000,
            'psi':x_['psi'] + h * x_['r'],
            'p':  x_['p'],
            'phi':x_['phi'] + h * x_['p'],
            'delta' : x_['delta'],
            'n' : x_['n']
            }
        return np.array([x__[var] for var in self.state_vec_order])