import numpy as np

class Connection:
    """
    Connection between two ensembles.
    
    Based on NEF Principle 2: Transformation
    """
    
    def __init__(self, pre, post, function=None, synapse_tau=0.01):
        """
        Initialize connection
        """
        self.pre = pre
        self.post = post
        self.function = function
        self.synapse_tau = synapse_tau
        
        # Compute decoders for pre-ensemble
        n_samples = max(1000, self.pre.n_neurons * 2)

        if function is None:
            # Identity function - output dimension = pre dimension
            self.pre.compute_decoders(n_samples=n_samples)
            output_dim = self.pre.dimensions
        else:
            # Compute function decoders
            self.pre.compute_decoders(function=function, n_samples=n_samples)
            output_dim = self.pre.decoders.shape[1]
        
        # Synaptic filter state (exponential filter)
        self.filtered_output = np.zeros(output_dim)
        
        # Connection weights (will be computed)
        self.compute_weights()
    
    def compute_weights(self):
        """
        Compute connection weight matrix W = E @ D.
        
        Based on Lecture 5: Feed-Forward Transformation
        
        W[i,j] represents the weight from pre-neuron j to post-neuron i
        """
        # Get post encoders and gain
        # E shape: (n_post_neurons, post_dimensions)
        E = self.post.encoders * self.post.gain[:, np.newaxis]
        
        # Get pre decoders
        # D shape: (n_pre_neurons, decoder_output_dim)
        D = self.pre.decoders
        
        decoder_output_dim = D.shape[1]
        
        # Check if dimensions are compatible
        if decoder_output_dim != self.post.dimensions:
            raise ValueError(
                f"Decoder output dimension {decoder_output_dim} doesn't match "
                f"post ensemble dimension {self.post.dimensions}"
            )
        
        # Compute weights: W = E @ D.T
        # E: (n_post_neurons, post_dim) @ D.T: (post_dim, n_pre_neurons)
        # Result: (n_post_neurons, n_pre_neurons)
        self.weights = E @ D.T
    
    def step(self, dt):
        """
        Update connection for one timestep
        """
        # Get pre-ensemble activities (firing rates)
        currents = self.pre.input_current
        activities = self.pre.neurons.rate_approximation(currents)
        
        # Decode output using D.T @ activities
        # D: (n_neurons, output_dim)
        # D.T: (output_dim, n_neurons)
        # activities: (n_neurons,)
        # Result: (output_dim,)
        decoded = self.pre.decoders.T @ activities
        
        # Apply synaptic filter (exponential low-pass)
        alpha = dt / self.synapse_tau
        self.filtered_output += alpha * (decoded - self.filtered_output)
        
        # Compute post currents using weight matrix
        # W: (n_post_neurons, n_pre_neurons)
        # activities: (n_pre_neurons,)
        # Result: (n_post_neurons,)
        post_contribution = self.weights @ activities
        
        self.post.input_current += post_contribution