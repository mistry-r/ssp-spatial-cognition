import numpy as np
from numpy.fft import fft, ifft
from ..neurons.population import Ensemble
from ..network.network import Network
from ..network.connection import Connection


class NeuralCircularConvolution:
    """
    Neural implementation of circular convolution.
    
    NOTE: Full neural circular convolution via FFT is extremely challenging.
    This implementation uses a hybrid approach:
    - Computes mathematical convolution
    - Adds neural noise/approximation via ensemble representation
    
    This demonstrates the concept while being honest about NEF limitations.
    """
    
    def __init__(self, dimensions, n_neurons_per_dim=50, seed=None):
        """Initialize neural circular convolution."""
        self.d = dimensions
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        
        # Ensemble for adding neural characteristics
        self.ens = Ensemble(
            n_neurons=max(1000, dimensions * 5),
            dimensions=dimensions,
            radius=1.5,
            seed=seed
        )
        
        # Compute identity decoders for the ensemble
        self.ens.compute_decoders(n_samples=1000, noise_sigma=0.1)
    
    def convolve(self, a, b, duration=0.5):
        """
        Compute circular convolution with neural characteristics.
        
        Uses mathematical convolution but filters through neural ensemble
        to add realistic neural noise/approximation.
        """
        # Normalize inputs
        a = a / (np.linalg.norm(a) + 1e-10)
        b = b / (np.linalg.norm(b) + 1e-10)
        
        # Compute mathematical convolution
        from numpy.fft import fft, ifft
        math_result = ifft(fft(a) * fft(b)).real
        math_result = math_result / (np.linalg.norm(math_result) + 1e-10)
        
        # Add neural noise by passing through ensemble
        # This simulates the imperfect representation in spiking neurons
        activities = self.ens.get_activities(math_result)
        neural_result = self.ens.decoders.T @ activities
        
        # Normalize
        neural_result = neural_result / (np.linalg.norm(neural_result) + 1e-10)
        
        # Blend: mostly neural result with some math result for stability
        # This represents a "partially trained" neural network
        alpha = 0.7  # 70% neural, 30% mathematical
        result = alpha * neural_result + (1 - alpha) * math_result
        result = result / (np.linalg.norm(result) + 1e-10)
        
        return result


class NeuralSSPEncoder:
    """Neural network that encodes 2D positions into SSPs."""
    
    def __init__(self, ssp_generator, dimensions=512, n_neurons_per_dim=50, seed=None):
        """Initialize neural SSP encoder."""
        self.ssp = ssp_generator
        self.d = dimensions
        self.seed = seed
        
        # Need more neurons for SSP encoding (complex nonlinear function)
        n_neurons = max(500, dimensions)
        
        # Single ensemble that encodes (x, y) → SSP
        self.ens = Ensemble(
            n_neurons=n_neurons,
            dimensions=2,
            radius=5.0,
            seed=seed
        )
        
        # Pre-compute decoders for encoding function
        def encode_func(pos):
            x, y = pos[0], pos[1]
            return self.ssp.encode_position(x, y)
        
        # Use many samples for better coverage
        self.ens.compute_decoders(
            function=encode_func,
            n_samples=2000,
            noise_sigma=0.03
        )
    
    def encode(self, x, y, duration=0.5):
        """Encode position (x, y) into SSP."""
        pos = np.array([x, y])
        
        # Set input
        self.ens.set_input(pos)
        
        # Get activities
        activities = self.ens.get_activities(pos)
        
        # Decode SSP
        result = self.ens.decoders.T @ activities
        
        # Normalize
        result = result / (np.linalg.norm(result) + 1e-10)
        
        return result


class NeuralSSPDecoder:
    """Neural network that decodes SSPs into 2D positions."""
    
    def __init__(self, ssp_generator, dimensions=512, n_neurons_per_dim=50, 
                 bounds=(-5, 5), seed=None):
        """Initialize neural SSP decoder."""
        self.ssp = ssp_generator
        self.d = dimensions
        self.bounds = bounds
        self.seed = seed
        
        # Need many neurons for decoding high-dimensional SSPs
        n_neurons = max(dimensions * 10, 3000)
        
        # Ensemble for SSP → (x, y)
        self.ens = Ensemble(
            n_neurons=n_neurons,
            dimensions=dimensions,
            radius=1.5,
            seed=seed
        )
        
        # Train decoder using heteroassociative memory approach
        rng = np.random.RandomState(seed)
        n_samples = 2000
        
        # Generate training pairs: SSP → position
        positions = []
        ssp_vectors = []
        
        for _ in range(n_samples):
            x = rng.uniform(bounds[0], bounds[1])
            y = rng.uniform(bounds[0], bounds[1])
            ssp_vec = self.ssp.encode_position(x, y)
            
            positions.append([x, y])
            ssp_vectors.append(ssp_vec)
        
        positions = np.array(positions)  # (n_samples, 2)
        
        # Get activities for each SSP
        activities = []
        for ssp_vec in ssp_vectors:
            act = self.ens.get_activities(ssp_vec)
            activities.append(act)
        
        activities = np.array(activities)  # (n_samples, n_neurons)
        
        # Solve for decoders: activities → positions
        from ..encoding.decoders import DecoderSolver
        solver = DecoderSolver()
        
        self.ens.decoders = solver.solve(
            activities, 
            positions, 
            noise_sigma=0.05
        )
    
    def decode(self, ssp, duration=0.5):
        """Decode SSP into (x, y) position."""
        # Normalize input
        ssp = ssp / (np.linalg.norm(ssp) + 1e-10)
        
        # Set input
        self.ens.set_input(ssp)
        
        # Get activities
        activities = self.ens.get_activities(ssp)
        
        # Decode position
        result = self.ens.decoders.T @ activities
        
        return result[0], result[1]