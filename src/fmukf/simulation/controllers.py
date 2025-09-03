import colorednoise
import numpy as np
from scipy.stats import norm


class ControllerBase:
    """Base class for controllers that generate input vectors for simulation environments.
    
    Controllers provide input vectors (u) based on current time and system state.
    Subclasses should implement the get_input method to define specific control behaviors.
    """
    
    def __init__(self, h: float):
        """Initialize the controller with sampling time parameter.
        
        Parameters
        ----------
        h : float
            Sampling time parameter, i.e., the amount of time between which a new input is required.
            With h=2.0, the controller will return a new input every 2.0 seconds.
        """
        self.h = h
        self.name = "MISSING NAME"
        self.description = "MISSING DESCRIPTION"

    def get_input(self, t: float, x: np.ndarray) -> np.ndarray:
        """Returns the output vector u for current state at time t.
        
        Parameters
        ----------
        t : float
            Current simulation time.
        x : np.ndarray
            Current state vector of the system.
            
        Returns
        -------
        np.ndarray
            Control input vector u.
        """
        raise NotImplementedError
        return u
    
    def __call__(self, t: float, x: np.ndarray) -> np.ndarray:
        """Syntactic sugar wrapper for the get_input method.
        
        Parameters
        ----------
        t : float
            Current simulation time.
        x : np.ndarray
            Current state vector of the system.
            
        Returns
        -------
        np.ndarray
            Control input vector u.
        """
        return self.get_input(t, x)


def uniform_colored_noise(Nsamples: int = 2**18, beta: float = 1, random_state = None) -> np.ndarray:
    """Generates uniformly distributed colored noise samples.
    
    Parameters
    ----------
    Nsamples : int
        Number of noise samples to generate.
    beta : float
        Determines the color by enforcing PSD shape of e^-f.
        Default: beta=1 (pink noise, i.e., 1/f noise).
    random_state : int or np.random.RandomState, optional
        Seed for random number generation.
        
    Returns
    -------
    np.ndarray
        Array of shape (Nsamples,) containing uniformly distributed colored noise in range (0,1).
    """
    # First generate coloured noise samples (should be gaussian)
    x = colorednoise.powerlaw_psd_gaussian(beta, Nsamples, random_state= random_state)
    
    x = (x - np.mean(x)) / np.std(x) # Scale to Normal Gaussian N(0,1)
    x = norm.cdf(x) # Probit transformation: Ie scales to uniform but retains time-correllation of n
    return x
    

class PinkNoiseController(ControllerBase):
    """Controller that generates uniform random pink noise actions.
    
    This controller produces colored noise control inputs with specified statistical properties.
    It pre-generates a queue of input vectors and returns them sequentially based on the
    sampling time parameter h.
    """
    
    def __init__(self, EnvSim: 'EnvSimBase', u_ranges = None, h: float = 0, Nsamples: int = 2**18, beta: float = 1., random_seed = None):
        """Initialize a pink noise controller.
        
        Parameters
        ----------
        EnvSim : EnvSimBase
            An environment simulator class, from whose parameters u_ranges can be automatically
            constructed instead of being manually specified.
        u_ranges : list of tuple
            The range of each variable in u in format [(min0, max0), (min1, max1),...(minN, maxN)].
            If None, then u_ranges is constructed from EnvSim parameters.
        h : float
            The minimum time (s) that needs to pass in order for .get_input() to return
            a new u vector from the queue.
        Nsamples : int
            The amount of u vectors to pregenerate (small numbers might mean noise is not
            correctly statistically distributed).
        beta : float
            Determines the e^-beta PSD color of the noise (default 1 ie pink ie 1/f noise).
        random_seed : int, optional
            Seed for random number generation.

        Example
            >>> from fmukf.simulation.container import Container
            >>> from fmukf.simulation.controllers import PinkNoiseController
            >>> env = Container()
            >>> x0 = env.sample_x0()
            >>> controller = PinkNoiseController(env, h=2.0, Nsamples=1000, beta=1.0, random_seed=42)
            >>> u0 = controller(0, x0)
        """

        if EnvSim is not None:
            u_ranges= [EnvSim.input_space[var]['sample_range'] for var in EnvSim.input_vec_order]
        elif u_ranges is None:
            raise ValueError("Both u_ranges and EnvSim are None. You at least need to specify one of them.")
        
        # Compute different value of beta for each variable
        # if type(beta) not in [list, dict]:
        #     beta = [beta for var in EnvSim.input_vec_order]
        # elif type(beta) == dict:
        #     beta = [beta[key] for key in EnvSim.input_vec_order]
          
        # Remember parameters
        self.h = h
        self.N = Nsamples
        self.u_ranges = u_ranges
        self.betas = [beta for var in EnvSim.input_vec_order]
        self.reset_rng(random_seed)

        # Pregenerate N input vectors with pink noise (ie array of shape [N,d])
        self.generate_u_queue()
        self.u = None         # Current u vector
        self.t_last = -np.inf # Last time (s) since a new u vector was was returned

        # Name and description
        self.name = "PinkNoiseController"
        self.description = f"""PinkNoiseController with delay h={h}s, beta={beta}, random_state={random_seed}, and uranges:\n{u_ranges}"""
  
    def reset_rng(self, random_state = None):
        """Reset the random number generator.
        
        Parameters
        ----------
        random_state : int, optional
            Seed for random number generation. If None, a random seed will be generated.
        """
        if random_state is None:
            import secrets
            random_state = secrets.randbits(32)
        self.rng = np.random.RandomState(random_state)
        self.generate_u_queue()

    def generate_u_queue(self):
        """Generate a queue of self.N pregenerated noise vectors.
        
        Creates self.u_queue with shape (self.N, d) where d is the dimension of the input space.
        Each row represents one input vector with colored noise properties.
        """
        d = len(self.u_ranges)
        self.u_queue=np.empty ((self.N, d))

        for idx, (min_val, max_val) in enumerate(self.u_ranges):
            seed = self.rng.randint(0,1e9)   #<-- Need random seed for each, otherwise correllated
            noise = uniform_colored_noise(self.N, self.betas[idx], random_state = seed) # Uniform in range (0,1)
            self.u_queue[:,idx] = min_val + noise * (max_val - min_val) 
        self.iterator = iter(self.u_queue)

    def get_input(self, t: float, x: np.ndarray) -> np.ndarray:
        """Get input vector u for current state at time t.
        
        Parameters
        ----------
        t : float
            Current simulation time.
        x : np.ndarray
            Current state vector (not used in this controller).
            
        Returns
        -------
        np.ndarray
            Control input vector u with shape (d,) where d is the input dimension.
        """
        # Get next u if enough time has passed (otherwise return previous u)
        if t >= self.t_last + self.h:
            try:
              self.u = next(self.iterator)
            except StopIteration:
            # If the queue is exhausted, generate a new one
              self.generate_u_queue()
              self.iterator = iter(self.u_queue)
              self.u = next(self.iterator)
            self.t_last = t
        return self.u