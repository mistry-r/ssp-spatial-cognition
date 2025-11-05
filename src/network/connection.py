import numpy as np

class Connection:
    """
    Connection between two ensembles.
    
    Based on NEF Principle 2: Transformation
    """
    
    def __init__(self, pre, post, function=None, synapse_tau=0.01):
        """
        Initialize connection.
        """
        self.pre = pre
        self.post = post
        self.function = function
        self.synapse_tau = synapse_tau
        
        # Compute decoders for this connection
        if function is None:
            self.pre.compute_decoders()
        else:
            self.pre.compute_decoders(function=function)
        
        # Synaptic filter state (exponential filter)
        self.filtered_output = np.zeros(self.pre.dimensions)
        
        # Connection weights
        self.compute_weights()
    
    def compute_weights(self):
        """
        Compute connection weight matrix W = E @ D.
        
        Based on Lecture 5: Feed-Forward Transformation
        """
        # Get post encoders and gain
        E = self.post.encoders * self.post.gain[:, np.newaxis]
        
        # Get pre decoders
        D = self.pre.decoders
        
        # Compute weights: W = E @ D
        self.weights = E @ D
    
    def step(self, dt):
        """
        Update connection for one timestep.
        """
        # Get pre-ensemble activities
        currents = self.pre.input_current
        activities = self.pre.neurons.rate_approximation(currents)
        
        # Decode output
        decoded = self.pre.decoders.T @ activities
        
        # Apply synaptic filter (exponential low-pass)
        # dx/dt = (1/tau) * (input - x)
        alpha = dt / self.synapse_tau
        self.filtered_output += alpha * (decoded - self.filtered_output)
        
        # Send to post-ensemble
        # Compute post currents: J = weights @ activities + bias
        post_currents = self.weights @ activities + self.post.bias
        self.post.input_current += post_currents