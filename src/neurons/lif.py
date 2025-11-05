import numpy as np

class LIFNeuron:
    """
    Leaky Integrate-and-Fire (LIF) neuron model.
    
    Based on Lecture 2: Neurons
    
    Dynamics (normalized):
        dv/dt = (1/τ_RC) * (-v + J)  if v < v_th
        v = 0 (reset) after spike, refractory period τ_ref
    
    Rate approximation:
        G[J] = 1 / (τ_ref - τ_RC * log(1 - 1/J))  if J > 1 = 0 otherwise
    """

    def __init__(self, tau_rc=0.02, tau_ref=0.002, v_th=1.0):
        """
        Initialize LIF neuron parameters.
        """
        self.tau_rc = tau_rc  # Membrane time constant
        self.tau_ref = tau_ref  # Refractory period
        self.v_th = v_th  # Spike threshold

        self.v = 0.0  # Membrane potential
        self.refractory_time = 0.0  # Time remaining in refractory period
    
    def step(self, J, dt):
        """
        Simulate one timestep of the LIF neuron
        """
        # Check if in refractory period
        if self.refractory_time > 0:
            self.refractory_time -= dt
            return False  # No spike
        
        # Update membrane potential
        dv = (dt / self.tau_rc) * (-self.v + J)
        self.v += dv

        # Check for spike
        if self.v >= self.v_th:
            self.v = 0.0  # Reset potential
            self.refractory_time = self.tau_ref  # Set refractory period
            return True  # Spike occurred
        
        return False  # No spike
    
    def rate_approximation(self, J):
        """
        Compute the firing rate approximation for a given input current J.
        """
        J = np.asarray(J)
        rate = np.zeros_like(J, dtype=float)

        # Only compute rate for J > 1 (above threshold)
        valid = J > 1.0

        if np.any(valid):
            rate[valid] = 1.0 / (
                self.tau_ref - self.tau_rc * np.log(1.0 - 1.0 / J[valid])
            )

        return rate
    
    def reset(self):
        """
        Reset the neuron state.
        """
        self.v = 0.0
        self.refractory_time = 0.0

class LIFPopulation:
    """
    Population of LIF neurons with shared parameters but independent state
    """

    def __init__(self, n_neurons, tau_rc=0.02, tau_ref=0.002):
        """
        Initialize a population of LIF neurons.
        """
        self.n_neurons = n_neurons
        self.tau_rc = tau_rc
        self.tau_ref = tau_ref

        self.v = np.zeros(n_neurons)
        self.refractory_time = np.zeros(n_neurons)

    def step(self, J, dt):
        """
        Simulate one timestep for the population of LIF neurons.
        """
        # Neurons in refractory period
        refractory = self.refractory_time > 0
        self.refractory_time[refractory] -= dt
        self.refractory_time = np.maximum(0, self.refractory_time)

        # Update membrane potentials
        active = ~refractory
        dv = np.zeros(self.n_neurons)
        active = ~refractory
        dv = np.zeros(self.n_neurons)
        dv[active] = (dt / self.tau_rc) * (-self.v[active] + J[active])
        self.v += dv
        
        # Check for spikes
        spikes = self.v >= 1.0
        
        # Reset spiked neurons
        self.v[spikes] = 0.0
        self.refractory_time[spikes] = self.tau_ref
        
        return spikes
    
    def rate_approximation(self, J):
        """
        Compute firing rates for all neurons.
        """
        rates = np.zeros(self.n_neurons)
        valid = J > 1.0
        
        if np.any(valid):
            rates[valid] = 1.0 / (
                self.tau_ref - self.tau_rc * np.log(1.0 - 1.0 / J[valid])
            )
        
        return rates
    
    def reset(self):
        """Reset all neuron states."""
        self.v = np.zeros(self.n_neurons)
        self.refractory_time = np.zeros(self.n_neurons)