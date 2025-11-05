import numpy as np
from .lif import LIFPopulation
from .tuning import generate_tuning_curves, compute_gain_bias, get_activities
from ..encoding.encoders import EncoderGenerator
from ..encoding.decoders import DecoderSolver

class Ensemble:
    """
    Neural ensemble representing a d-dimensional value.
    
    Based on NEF Principle 1: Representation
    """
    
    def __init__(self, n_neurons, dimensions, radius=1.0, 
                 tau_rc=0.02, tau_ref=0.002, seed=None):
        """
        Initialize neural ensemble.
        """
        self.n_neurons = n_neurons
        self.dimensions = dimensions
        self.radius = radius
        self.tau_rc = tau_rc
        self.tau_ref = tau_ref
        
        # Random number generator
        self.rng = np.random.RandomState(seed)
        
        # Generate tuning curves
        self.encoders, max_rates, intercepts = generate_tuning_curves(
            n_neurons, dimensions, self.rng
        )
        
        # Compute gain and bias
        self.gain, self.bias = compute_gain_bias(
            self.encoders, max_rates, intercepts, radius
        )
        
        # LIF neuron population
        self.neurons = LIFPopulation(n_neurons, tau_rc, tau_ref)
        
        # Decoders (computed later)
        self.decoders = None
        
        # Current input
        self.input_current = np.zeros(n_neurons)
        
    def get_activities(self, x):
        """
        Get neural activities (firing rates) for input x.
        """
        return get_activities(x, self.encoders, self.gain, self.bias)
    
    def compute_decoders(self, n_samples=500, noise_sigma=0.1, function=None):
        """
        Compute optimal decoders for this ensemble.
        """
        # Generate random samples from represented space
        x_samples = self.rng.uniform(-self.radius, self.radius, 
                                     size=(n_samples, self.dimensions))
        
        # Compute activities for each sample
        activities = np.array([self.get_activities(x) for x in x_samples]).T
        
        # Solve for decoders
        solver = DecoderSolver()
        
        if function is None:
            # Identity decoder
            targets = x_samples.T
            self.decoders = solver.solve(activities, targets, noise_sigma)
        else:
            # Function decoder
            self.decoders = solver.solve_function(
                activities, x_samples, function, noise_sigma
            )
        
        return self.decoders
    
    def decode(self, activities=None):
        """
        Decode represented value from neural activities.
        """
        if self.decoders is None:
            raise ValueError("Decoders not computed. Call compute_decoders() first.")
        
        if activities is None:
            # Use current firing rates
            currents = self.input_current
            activities = self.neurons.rate_approximation(currents)
        
        return self.decoders.T @ activities
    
    def step(self, dt):
        """
        Simulate one timestep.
        """
        return self.neurons.step(self.input_current, dt)
    
    def set_input(self, x):
        """
        Set input value to be represented.
        """
        # Compute input currents: J = alpha * <x, e> + J_bias
        self.input_current = self.gain * (x @ self.encoders.T) + self.bias
    
    def reset(self):
        """Reset ensemble state."""
        self.neurons.reset()
        self.input_current = np.zeros(self.n_neurons)