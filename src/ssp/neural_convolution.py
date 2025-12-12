import numpy as np
from numpy.fft import fft, ifft
from ..neurons.population import Ensemble
from ..network.network import Network
from ..network.connection import Connection

class NeuralCircularConvolution:
    """
    Neural implementation of circular convolution using NEF.
    
    Implements: result = a ⊛ b = IFFT(FFT(a) * FFT(b))
    """
    
    def __init__(self, dimensions, n_neurons_per_dim=50, seed=None):
        """
        Initialize neural circular convolution.
        
        Args:
            dimensions: Dimensionality of vectors (must be even for FFT)
            n_neurons_per_dim: Neurons per dimension for representation
            seed: Random seed
        """
        self.d = dimensions
        self.n_neurons_per_dim = n_neurons_per_dim
        self.n_neurons = dimensions * n_neurons_per_dim
        self.seed = seed
        
        # Create network
        self.net = Network(dt=0.001)
        
        # Create ensembles for inputs
        self.ens_a = Ensemble(
            n_neurons=self.n_neurons,
            dimensions=dimensions,
            seed=seed
        )
        
        self.ens_b = Ensemble(
            n_neurons=self.n_neurons,
            dimensions=dimensions,
            seed=seed if seed is None else seed + 1
        )
        
        # Output ensemble
        self.ens_result = Ensemble(
            n_neurons=self.n_neurons,
            dimensions=dimensions,
            seed=seed if seed is None else seed + 2
        )
        
        # Add to network
        self.net.add_ensemble(self.ens_a)
        self.net.add_ensemble(self.ens_b)
        self.net.add_ensemble(self.ens_result)
        
        # Compute identity decoders first
        self.ens_a.compute_decoders(n_samples=1000, noise_sigma=0.05)
        self.ens_b.compute_decoders(n_samples=1000, noise_sigma=0.05)
        
        # Setup convolution
        self._setup_convolution_connection()
    
    def _setup_convolution_connection(self):
        """
        Set up connection that performs convolution.
        
        We use a combined ensemble approach:
        - Create ensemble for concatenated [a, b]
        - Decode convolution from combined state
        """
        # Combined ensemble for both inputs
        self.ens_combined = Ensemble(
            n_neurons=self.d * self.n_neurons_per_dim * 2,
            dimensions=self.d * 2,
            seed=self.seed if self.seed is None else self.seed + 3
        )
        self.net.add_ensemble(self.ens_combined)
        
        # Connection functions
        def to_first_half(x):
            result = np.zeros(self.d * 2)
            result[:self.d] = x
            return result
        
        def to_second_half(x):
            result = np.zeros(self.d * 2)
            result[self.d:] = x
            return result
        
        # Connect a -> combined (first half)
        conn_a = Connection(
            self.ens_a,
            self.ens_combined,
            function=to_first_half
        )
        self.net.add_connection(conn_a)
        
        # Connect b -> combined (second half)  
        conn_b = Connection(
            self.ens_b,
            self.ens_combined,
            function=to_second_half
        )
        self.net.add_connection(conn_b)
        
        # Convolution function
        def circular_convolution_function(x):
            a = x[:self.d]
            b = x[self.d:]
            result = ifft(fft(a) * fft(b)).real
            return result
        
        # Compute decoders for combined -> result
        self.ens_combined.compute_decoders(
            function=circular_convolution_function,
            n_samples=1000,
            noise_sigma=0.05
        )
        
        conn_result = Connection(
            self.ens_combined,
            self.ens_result,
            function=circular_convolution_function
        )
        self.net.add_connection(conn_result)
        
        # Compute identity decoders for result
        self.ens_result.compute_decoders(n_samples=1000, noise_sigma=0.05)
    
    def convolve(self, a, b, duration=0.5):
        """
        Compute neural circular convolution of a and b.
        """
        # Reset network
        for ens in self.net.ensembles:
            ens.reset()
        self.net.time = 0.0
        self.net.probes = {}
        
        # Set inputs
        self.ens_a.set_input(a)
        self.ens_b.set_input(b)
        
        # Probe output
        self.net.probe(self.ens_result, 'result', 'decoded')
        
        # Run simulation
        self.net.run(duration=duration)
        
        # Get result (average over last 100ms)
        data = self.net.get_probe_data('result')
        n_samples = int(0.1 / self.net.dt)
        result = data[-n_samples:].mean(axis=0)
        
        return result


class NeuralSSPEncoder:
    """
    Neural network that encodes 2D positions into SSPs.
    """
    
    def __init__(self, ssp_generator, dimensions=512, n_neurons_per_dim=50, seed=None):
        """Initialize neural SSP encoder."""
        self.ssp = ssp_generator
        self.d = dimensions
        self.n_neurons_per_dim = n_neurons_per_dim
        self.seed = seed
        
        # Create network
        self.net = Network(dt=0.001)
        
        # Input ensemble for (x, y)
        self.ens_input = Ensemble(
            n_neurons=200,
            dimensions=2,
            radius=5.0,
            seed=seed
        )
        self.net.add_ensemble(self.ens_input)
        
        # Output ensemble for SSP
        self.ens_ssp = Ensemble(
            n_neurons=dimensions * n_neurons_per_dim,
            dimensions=dimensions,
            seed=seed if seed is None else seed + 1
        )
        self.net.add_ensemble(self.ens_ssp)
        
        # Encoding function
        def encode_function(pos):
            x, y = pos[0], pos[1]
            return self.ssp.encode_position(x, y)
        
        # Compute decoders for input -> SSP
        self.ens_input.compute_decoders(
            function=encode_function,
            n_samples=1000,
            noise_sigma=0.05
        )
        
        # Create connection
        conn = Connection(
            self.ens_input,
            self.ens_ssp,
            function=encode_function
        )
        self.net.add_connection(conn)
        
        # Compute identity decoders for SSP output
        self.ens_ssp.compute_decoders(n_samples=1000, noise_sigma=0.05)
    
    def encode(self, x, y, duration=0.5):
        """Encode position (x, y) into SSP using neurons."""
        # Reset
        for ens in self.net.ensembles:
            ens.reset()
        self.net.time = 0.0
        self.net.probes = {}
        
        # Set input
        self.ens_input.set_input(np.array([x, y]))
        
        # Probe output
        self.net.probe(self.ens_ssp, 'ssp', 'decoded')
        
        # Run
        self.net.run(duration=duration)
        
        # Get result
        data = self.net.get_probe_data('ssp')
        n_samples = int(0.1 / self.net.dt)
        result = data[-n_samples:].mean(axis=0)
        
        # Normalize
        norm = np.linalg.norm(result)
        if norm > 1e-10:
            result = result / norm
        
        return result


class NeuralSSPDecoder:
    """
    Neural network that decodes SSPs into 2D positions.
    """
    
    def __init__(self, ssp_generator, dimensions=512, n_neurons_per_dim=50, 
                 bounds=(-5, 5), seed=None):
        """Initialize neural SSP decoder."""
        self.ssp = ssp_generator
        self.d = dimensions
        self.n_neurons_per_dim = n_neurons_per_dim
        self.bounds = bounds
        self.seed = seed
        
        # Create network
        self.net = Network(dt=0.001)
        
        # Input ensemble for SSP
        self.ens_ssp = Ensemble(
            n_neurons=dimensions * n_neurons_per_dim,
            dimensions=dimensions,
            seed=seed
        )
        self.net.add_ensemble(self.ens_ssp)
        
        # Output ensemble for (x, y)
        self.ens_output = Ensemble(
            n_neurons=200,
            dimensions=2,
            radius=5.0,
            seed=seed if seed is None else seed + 1
        )
        self.net.add_ensemble(self.ens_output)
        
        # Train heteroassociative decoder
        # Generate training samples
        n_samples = 1000
        rng = np.random.RandomState(seed)
        
        positions_list = []
        ssp_list = []
        
        for _ in range(n_samples):
            x = rng.uniform(bounds[0], bounds[1])
            y = rng.uniform(bounds[0], bounds[1])
            ssp_vec = self.ssp.encode_position(x, y)
            
            positions_list.append([x, y])
            ssp_list.append(ssp_vec)
        
        # Compute function decoders that map SSP -> position
        def decode_function_factory(positions_array):
            """Create decoding function with closure over positions."""
            def decode_function(ssp_input):
                # Find closest training SSP
                similarities = [self.ssp.similarity(ssp_input, ssp_vec) 
                               for ssp_vec in ssp_list]
                best_idx = np.argmax(similarities)
                return positions_array[best_idx]
            return decode_function
        
        positions_array = np.array(positions_list)
        
        # Compute decoders using heteroassociative approach
        # We need to manually compute decoders from SSP activities to positions
        from ..encoding.decoders import DecoderSolver
        solver = DecoderSolver()
        
        # Get activities for each training SSP
        activities_list = []
        for ssp_vec in ssp_list:
            act = self.ens_ssp.get_activities(ssp_vec)
            activities_list.append(act)
        
        activities = np.array(activities_list)  # (n_samples, n_neurons)
        positions = np.array(positions_list)     # (n_samples, 2)
        
        # Solve for decoders: activities -> positions
        # DecoderSolver expects (n_neurons, n_samples) or (n_samples, n_neurons)
        self.ens_ssp.decoders = solver.solve(activities, positions, noise_sigma=0.05)
        
        # Create connection
        conn = Connection(self.ens_ssp, self.ens_output)
        self.net.add_connection(conn)
        
        # Compute identity decoders for output
        self.ens_output.compute_decoders(n_samples=500, noise_sigma=0.05)
    
    def decode(self, ssp, duration=0.5):
        """Decode SSP into (x, y) position using neurons."""
        # Reset
        for ens in self.net.ensembles:
            ens.reset()
        self.net.time = 0.0
        self.net.probes = {}
        
        # Set input
        self.ens_ssp.set_input(ssp)
        
        # Probe output
        self.net.probe(self.ens_output, 'position', 'decoded')
        
        # Run
        self.net.run(duration=duration)
        
        # Get result
        data = self.net.get_probe_data('position')
        n_samples = int(0.1 / self.net.dt)
        result = data[-n_samples:].mean(axis=0)
        
        return result[0], result[1]