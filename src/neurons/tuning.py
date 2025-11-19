import numpy as np
from .lif import LIFNeuron

def generate_tuning_curves(n_neurons, dimensions, rng=None):
    """
    Generate random tuning curve parameters for a neural ensemble
    """
    if rng is None:
        rng = np.random.RandomState()
    
    # Generate encoders (preferred directions)
    encoders = rng.randn(n_neurons, dimensions)
    encoders /= np.linalg.norm(encoders, axis=1, keepdims=True)
    
    # Generate maximum firing rates
    max_rates = rng.uniform(100, 200, size=n_neurons) # Hz
    
    # Generate intercepts
    intercepts = rng.uniform(-1, 1, size=n_neurons)
    
    return encoders, max_rates, intercepts

def compute_gain_bias(encoders, max_rates, intercepts, radius=1.0):
    """
    Compute gain (alpha) and bias (J_bias) from tuning curve parameters

    Based on Lecture 3: Representation
    
    The encoding equation is:
        J_i = alpha_i * <x, e_i> + J_bias_i
    
    We want:
        - At intercept: J_i = 1 (threshold)
        - At max point: firing rate = max_rate
    """
    n_neurons = len(encoders)

    # Compute the current needed for max firing rate
    # Using inverse of LIF rate equation: J = 1 / (1 - exp(-1/(rate * tau_ref)))
    tau_rc = 0.02  # Membrane time constant
    tau_ref = 0.002  # Refractory period

    # Approximate: for high rates, J ≈ rate * (tau_ref + tau_rc)
    # More accurate: solve J from rate equation
    max_current = np.zeros(n_neurons)
    for i, rate in enumerate(max_rates):
        if rate > 0:
            # Solve: rate = 1 / (tau_ref - tau_rc * log(1 - 1/J))
            # Rearranged: J = 1 / (1 - exp(-(1/rate - tau_ref)/tau_rc))
            max_current[i] = 1.0 / (1.0 - np.exp(-(1.0/rate - tau_ref) / tau_rc))
    
    # Solve linear system:
    # 1 = alpha * intercept + J_bias
    # max_current = alpha * radius + J_bias
    
    gain = (max_current - 1.0) / (radius - intercepts)
    bias = 1.0 - gain * intercepts
    
    return gain, bias

def get_activities(x, encoders, gain, bias):
    """
    Compute neural activities (firing rates) for input x

    Based on Lecture 3: Representation
    
    J_i = alpha_i * <x, e_i> + J_bias_i
    a_i = G[J_i]  (LIF rate approximation)
    """
    x = np.atleast_2d(x)
    n_samples = x.shape[0]
    n_neurons = len(encoders)
    
    # Compute currents: J = alpha * <x, e> + J_bias
    # Shape: (n_samples, n_neurons)
    currents = gain * (x @ encoders.T) + bias
    
    # Compute firing rates using LIF rate approximation
    neuron = LIFNeuron()
    activities = neuron.rate_approximation(currents)
    
    if n_samples == 1:
        return activities[0]
    return activities
