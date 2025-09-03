import torch

def correct_positions_with_integration(
    x: torch.Tensor,
    x_pred: torch.Tensor,
    h: float,
    method: str = 'midpoint'
) -> torch.Tensor:
    """
    Post-processes a neural network's predicted next state to reduce noise in 
    position-like features by numerically integrating velocity-like features 
    over the time step.

    This function addresses a common issue in learned transition models used 
    in filtering and simulation: the model may predict velocity-related states 
    accurately (e.g., surge/sway speeds, angular rates) but produce noisy 
    position-related states (e.g., x, y, orientation angles) when unrolled. 
    To mitigate this, the position-like features in `x_pred` are replaced 
    with values obtained by integrating the corresponding velocity features 
    from `x` and/or `x_pred` using a specified integration scheme 
    (e.g., forward Euler, trapezoidal, midpoint).

    Integration is performed over the interval [t, t+h], using the state vector 
    ordering defined in `fmukf.simulation.container.ScaledContainer`. The 
    mapping from velocity to position is:
        - North/East positions (x, y) ← body-frame surge/sway velocities (u, v) 
          transformed using yaw angle `psi`
        - Yaw angle (psi) ← yaw rate r
        - Roll angle (phi) ← roll rate p


    Note: Currenlty hard-coded for the reference frame / state space definition
          for the fmfukf.simulation.container.ScaledContainer.

    Parameters
    ----------
    x : torch.Tensor
        State at the current time t, shape (..., 10).
    x_pred : torch.Tensor
        Neural network predicted state at time t+h, shape (..., 10).
    h : float
        Time step in seconds.
    method : str, optional
        Numerical integration method to use for position correction:
        - 'none' : return x_pred without modification
        - 'forward_euler_current' : integrate using current velocities (from x)
        - 'forward_euler_predicted' : integrate using predicted velocities (from x_pred)
        - 'trapezoidal' : integrate using average of current and predicted velocities
        - 'midpoint' : integrate using midpoint velocity estimate
        Default is 'trapezoidal'.
    Returns
    -------
    torch.Tensor
        Corrected prediction, same shape as x_pred, on the same device/dtype as x_pred.
    """

    device = x_pred.device
    dtype = x_pred.dtype

    state_vec_order = ['u','v','r','x','y','psi','p','phi','delta','n']
    # Convert h into a tensor on the correct device & dtype
    h_t = torch.tensor(h, device=device, dtype=dtype)

    # Index dictionary, e.g. var_idx['x'] = 3 if 'x' is the 4th in state_vec_order
    var_idx = {var: i for i, var in enumerate(state_vec_order)}

    # We'll modify a copy of x_pred rather than in-place
    x_corrected = x_pred.clone()

    # Helper to compute velocities from a tensor with shape (..., 10)
    def get_velocities(X: torch.Tensor):
        """
        Given a state tensor X (...,10), extract velocity components:
          - Vn, Ve in m/s (north/east)
          - r, p in deg/s (yaw rate, roll rate)
        """
        psi_rad = torch.deg2rad(X[..., var_idx['psi']])
        u = X[..., var_idx['u']]
        v = X[..., var_idx['v']]
        r = X[..., var_idx['r']]
        p = X[..., var_idx['p']]

        # Compute north/east velocity
        Vn = u * torch.cos(psi_rad) - v * torch.sin(psi_rad)
        Ve = u * torch.sin(psi_rad) + v * torch.cos(psi_rad)
        return Vn, Ve, r, p

    # 1. Extract velocities at the current state and predicted next state
    Vn_curr, Ve_curr, r_curr, p_curr = get_velocities(x)
    Vn_pred, Ve_pred, r_pred, p_pred = get_velocities(x_pred)

    # 2. Based on the chosen method, correct the position-like variables
    if method == 'none':
        # do not correct anything, just return x_pred
        return x_pred

    elif method == 'forward_euler_current':
        # Use only the *current* velocities for the entire step
        x_corrected[..., var_idx['x']]   = x[..., var_idx['x']]   + (h_t / 1000.0) * Vn_curr
        x_corrected[..., var_idx['y']]   = x[..., var_idx['y']]   + (h_t / 1000.0) * Ve_curr
        x_corrected[..., var_idx['psi']] = x[..., var_idx['psi']] + h_t * r_curr
        x_corrected[..., var_idx['phi']] = x[..., var_idx['phi']] + h_t * p_curr

    elif method == 'forward_euler_predicted':
        # Use only the *predicted* velocities for the entire step
        x_corrected[..., var_idx['x']]   = x[..., var_idx['x']]   + (h_t / 1000.0) * Vn_pred
        x_corrected[..., var_idx['y']]   = x[..., var_idx['y']]   + (h_t / 1000.0) * Ve_pred
        x_corrected[..., var_idx['psi']] = x[..., var_idx['psi']] + h_t * r_pred
        x_corrected[..., var_idx['phi']] = x[..., var_idx['phi']] + h_t * p_pred

    elif method == 'trapezoidal':
        # Average the two velocity estimates
        x_corrected[..., var_idx['x']]   = x[..., var_idx['x']]   + (h_t / 1000.0) * 0.5 * (Vn_curr + Vn_pred)
        x_corrected[..., var_idx['y']]   = x[..., var_idx['y']]   + (h_t / 1000.0) * 0.5 * (Ve_curr + Ve_pred)
        x_corrected[..., var_idx['psi']] = x[..., var_idx['psi']] + h_t * 0.5 * (r_curr + r_pred)
        x_corrected[..., var_idx['phi']] = x[..., var_idx['phi']] + h_t * 0.5 * (p_curr + p_pred)

    elif method == 'midpoint':
        # Midpoint method with approximate midpoint velocity
        Vn_mid = 0.5 * (Vn_curr + Vn_pred)
        Ve_mid = 0.5 * (Ve_curr + Ve_pred)
        r_mid  = 0.5 * (r_curr + r_pred)
        p_mid  = 0.5 * (p_curr + p_pred)

        x_corrected[..., var_idx['x']]   = x[..., var_idx['x']]   + (h_t / 1000.0) * Vn_mid
        x_corrected[..., var_idx['y']]   = x[..., var_idx['y']]   + (h_t / 1000.0) * Ve_mid
        x_corrected[..., var_idx['psi']] = x[..., var_idx['psi']] + h_t * r_mid
        x_corrected[..., var_idx['phi']] = x[..., var_idx['phi']] + h_t * p_mid

    else:
        raise ValueError(
            f"Unknown method '{method}'. "
            "Must be one of 'none', 'forward_euler_current', "
            "'forward_euler_predicted', 'trapezoidal', or 'midpoint'."
        )

    return x_corrected
