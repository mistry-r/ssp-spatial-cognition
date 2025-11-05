import numpy as np

class EncoderGenerator:
    """
    Generates encoders for neural ensembles
    """
    
    def __init__(self, seed=None):
        """
        Initialize encoder generator
        """
        self.rng = np.random.RandomState(seed)
    
    def generate_encoders(self, n_neurons, dimensions, encoder_type='random'):
        """
        Generate encoder vectors
        """
        if encoder_type == 'random':
            return self._random_encoders(n_neurons, dimensions)
        elif encoder_type == 'grid':
            return self._grid_encoders(n_neurons, dimensions)
        elif encoder_type == 'axis_aligned':
            return self._axis_aligned_encoders(n_neurons, dimensions)
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")
    
    def _random_encoders(self, n_neurons, dimensions):
        """
        Random unit vectors (most common)
        """
        encoders = self.rng.randn(n_neurons, dimensions)
        encoders /= np.linalg.norm(encoders, axis=1, keepdims=True)
        return encoders
    
    def _grid_encoders(self, n_neurons, dimensions):
        """
        Evenly spaced on unit sphere (2D only).
        """
        if dimensions != 2:
            raise ValueError("Grid encoders only implemented for 2D")
        
        angles = np.linspace(0, 2*np.pi, n_neurons, endpoint=False)
        encoders = np.column_stack([np.cos(angles), np.sin(angles)])
        return encoders
    
    def _axis_aligned_encoders(self, n_neurons, dimensions):
        """
        Aligned with coordinate axes (like vestibular system).
        """
        encoders = []
        
        # Positive and negative directions for each axis
        for dim in range(dimensions):
            for sign in [1, -1]:
                e = np.zeros(dimensions)
                e[dim] = sign
                encoders.append(e)
                
                if len(encoders) >= n_neurons:
                    break
            if len(encoders) >= n_neurons:
                break
        
        # Fill remaining with random if needed
        while len(encoders) < n_neurons:
            e = self.rng.randn(dimensions)
            e /= np.linalg.norm(e)
            encoders.append(e)
        
        return np.array(encoders[:n_neurons])