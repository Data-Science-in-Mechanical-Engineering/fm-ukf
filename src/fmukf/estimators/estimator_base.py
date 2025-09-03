import numpy as np


class EstimatorBase:
    """Base class for state estimators that process measurement and control data.
    
    State estimators provide state estimates (xhat) based on current measurements (y) 
    and control inputs (u). Subclasses should implement the __call__ method to define 
    specific estimation behaviors.
    
    The estimator follows a stateful pattern where:
    1. Estimator object is created once at the beginning of the trajectory simulation
    2. Estimator maintains internal state (covariance, history, etc.)
    3. Each call to __call__(y_k, u_k) assumes the next time-step is t_k+1 = t_k + h 
       and returns estimate xhat_k+1
    
    The simulation loop typically follows:
        y_k    = sensor(x_k, u_(k-1)) 
        xhat_k = estimator(y_k, u_(k-1)) 
        u_k    = controller(xhat_k)
        x_(k+1) = sim_step(x_k, u_k, h)
    
    Example usage:
        # Create estimator with initial parameters
        estimator = MyEstimator(EnvSim, Sensor, xhat0, P0, Q, h)
        

    """
    
    def __call__(self, y: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Returns the state estimate xhat for current measurement y and control input u.
                
        This method processes the next time step of measurements and control inputs
        to update the internal state and return the current state estimate.

        Each call assumes the next time-step is tracjed
        
        Parameters
        ----------
        y : np.ndarray
            Current measurement vector from sensors.
        u : np.ndarray
            Current control input vector.
            
        Returns
        -------
        np.ndarray
            State estimate vector xhat.
        """
        raise NotImplementedError
        return xhat