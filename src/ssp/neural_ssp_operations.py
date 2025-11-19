"""
Neural SSP Operations

Implements SSP operations (circular convolution, encoding, decoding) 
using spiking neural networks with the Neural Engineering Framework (NEF)
"""

import numpy as np
from ..neurons.population import Ensemble
from ..network.network import Network
from ..network.connection import Connection
from .fractional_binding import SpatialSemanticPointer

class NeuralCircularConvolution:
    """
    Neural implementation of circular convolution using FFT method.
    
    Computes: a ⊛ b = F^{-1}{F{a} ⊙ F{b}}
    
    This is implemented by:
    1. Computing FFT of both inputs (can be done with neurons or precomputed)
    2. Element-wise multiplication in frequency domain
    3. Computing IFFT to get result
    """
    
    def __init__(self, dimensions, n_neurons_per_dim=50, seed=None):
        """
        Initialize neural circular convolution.
        """
        self.d = dimensions
        self.n_neurons_per_dim = n_neurons_per_dim
        self.seed = seed
        
        # Total neurons needed
        self.n_neurons = dimensions * n_neurons_per_dim
        
        # Create ensemble to represent the convolution result
        self.result_ensemble = Ensemble(
            n_neurons=self.n_neurons,
            dimensions=dimensions,
            seed=seed
        )
    
    def create_network(self, input_a_dims, input_b_dims):
        """
        Create a network that computes circular convolution.
        """
        net = Network(dt=0.001)
        
        # Input ensembles
        ens_a = Ensemble(
            n_neurons=input_a_dims * self.n_neurons_per_dim,
            dimensions=input_a_dims,
            seed=self.seed
        )
        ens_b = Ensemble(
            n_neurons=input_b_dims * self.n_neurons_per_dim,
            dimensions=input_b_dims,
            seed=self.seed + 1 if self.seed else None
        )
        
        # Add to network
        net.add_ensemble(ens_a)
        net.add_ensemble(ens_b)
        net.add_ensemble(self.result_ensemble)
        
        return net, ens_a, ens_b
    
    def convolve_with_fixed(self, fixed_vector, variable_ensemble, output_ensemble):
        """
        Convolve a fixed vector with a variable vector represented by neurons.
        """
        # Precompute FFT of fixed vector
        fft_fixed = np.fft.fft(fixed_vector)
        
        def convolution_function(variable_vector):
            """Compute convolution in frequency domain"""
            # FFT of variable vector
            fft_variable = np.fft.fft(variable_vector)
            
            # Element-wise multiplication
            fft_result = fft_fixed * fft_variable
            
            # IFFT to get result
            result = np.fft.ifft(fft_result).real
            
            return result
        
        # Create connection that computes this function
        conn = Connection(
            variable_ensemble,
            output_ensemble,
            function=convolution_function
        )
        
        return conn


class NeuralSSPEncoder:
    """
    Neural network that encodes 2D positions into SSPs.
    
    Takes (x, y) coordinates as input and produces S(x,y) = X^x ⊛ Y^y as output.
    """
    
    def __init__(self, ssp_generator, n_neurons_per_dim=100, seed=None):
        """
        Initialize neural SSP encoder
        """
        self.ssp = ssp_generator
        self.d = ssp_generator.d
        self.n_neurons_per_dim = n_neurons_per_dim
        self.seed = seed
    
    def create_network(self):
        """
        Create a neural network that encodes positions
        """
        net = Network(dt=0.001)
        
        # Input ensemble for (x, y) coordinates
        input_ens = Ensemble(
            n_neurons=2 * self.n_neurons_per_dim,
            dimensions=2,
            radius=5.0,  # Assuming positions in [-5, 5]
            seed=self.seed
        )
        
        # Output ensemble for SSP
        output_ens = Ensemble(
            n_neurons=self.d * self.n_neurons_per_dim,
            dimensions=self.d,
            seed=self.seed + 1 if self.seed else None
        )
        
        # Connection that computes encoding
        def encoding_function(pos):
            x, y = pos
            return self.ssp.encode_position(x, y)
        
        conn = Connection(
            input_ens,
            output_ens,
            function=encoding_function
        )
        
        net.add_ensemble(input_ens)
        net.add_ensemble(output_ens)
        net.add_connection(conn)
        
        return net, input_ens, output_ens
    
    def create_separate_axis_network(self):
        """
        Alternative implementation: encode X and Y axes separately, then convolve.
        
        This is more modular and closer to how the brain might work.
        
        Returns
        -------
        net : Network
        input_ens : Ensemble for (x, y)
        x_ssp_ens : Ensemble for X^x
        y_ssp_ens : Ensemble for Y^y  
        output_ens : Ensemble for X^x ⊛ Y^y
        """
        net = Network(dt=0.001)
        
        # Input ensemble
        input_ens = Ensemble(
            n_neurons=2 * self.n_neurons_per_dim,
            dimensions=2,
            radius=5.0,
            seed=self.seed
        )
        
        # Separate ensembles for X^x and Y^y
        x_ssp_ens = Ensemble(
            n_neurons=self.d * self.n_neurons_per_dim,
            dimensions=self.d,
            seed=self.seed + 1 if self.seed else None
        )
        
        y_ssp_ens = Ensemble(
            n_neurons=self.d * self.n_neurons_per_dim,
            dimensions=self.d,
            seed=self.seed + 2 if self.seed else None
        )
        
        # Output ensemble for convolution result
        output_ens = Ensemble(
            n_neurons=self.d * self.n_neurons_per_dim,
            dimensions=self.d,
            seed=self.seed + 3 if self.seed else None
        )
        
        # Connection: input -> X^x
        def encode_x(pos):
            x = pos[0]
            return self.ssp.fractional_power(self.ssp.X, x)
        
        conn_x = Connection(input_ens, x_ssp_ens, function=encode_x)
        
        # Connection: input -> Y^y
        def encode_y(pos):
            y = pos[1]
            return self.ssp.fractional_power(self.ssp.Y, y)
        
        conn_y = Connection(input_ens, y_ssp_ens, function=encode_y)
        
        # Connection: (X^x, Y^y) -> X^x ⊛ Y^y
        def convolve_xy(combined):
            # Split the combined vector
            x_ssp = combined[:self.d]
            y_ssp = combined[self.d:]
            return self.ssp.circular_convolution(x_ssp, y_ssp)
        
        # Create a combined ensemble (this is a simplification)
        combined_ens = Ensemble(
            n_neurons=2 * self.d * self.n_neurons_per_dim,
            dimensions=2 * self.d,
            seed=self.seed + 4 if self.seed else None
        )
        
        # Note: This is a simplified architecture
        # A full implementation would use a proper multi-input circular convolution
        
        net.add_ensemble(input_ens)
        net.add_ensemble(x_ssp_ens)
        net.add_ensemble(y_ssp_ens)
        net.add_ensemble(combined_ens)
        net.add_ensemble(output_ens)
        net.add_connection(conn_x)
        net.add_connection(conn_y)
        
        return net, input_ens, x_ssp_ens, y_ssp_ens, output_ens


class NeuralSSPDecoder:
    """
    Neural network that decodes SSPs back to 2D positions.
    
    Uses a heteroassociative memory approach to map from 512-D SSP to (x, y).
    """
    
    def __init__(self, ssp_generator, n_neurons_per_dim=50, bounds=(-5, 5), seed=None):
        """
        Initialize neural SSP decoder
        """
        self.ssp = ssp_generator
        self.d = ssp_generator.d
        self.n_neurons_per_dim = n_neurons_per_dim
        self.bounds = bounds
        self.seed = seed
        
        # Train a decoder to map SSP -> (x, y)
        # This requires generating training data
        self.training_data = None
    
    def generate_training_data(self, n_samples=1000):
        """
        Generate training pairs of (SSP, position)
        """
        rng = np.random.RandomState(self.seed)
        
        positions = rng.uniform(
            self.bounds[0], 
            self.bounds[1], 
            size=(n_samples, 2)
        )
        
        ssps = np.array([
            self.ssp.encode_position(x, y) 
            for x, y in positions
        ])
        
        self.training_data = (ssps, positions)
        return ssps, positions
    
    def create_network(self):
        """
        Create a neural network that decodes SSPs to positions
        """
        # Generate training data if not already done
        if self.training_data is None:
            self.generate_training_data()
        
        ssps, positions = self.training_data
        
        net = Network(dt=0.001)
        
        # Input ensemble for SSP
        input_ens = Ensemble(
            n_neurons=self.d * self.n_neurons_per_dim,
            dimensions=self.d,
            seed=self.seed
        )
        
        # Output ensemble for (x, y)
        output_ens = Ensemble(
            n_neurons=2 * self.n_neurons_per_dim * 5,  # Extra neurons for better accuracy
            dimensions=2,
            radius=np.max(np.abs(self.bounds)) * 1.5,
            seed=self.seed + 1 if self.seed else None
        )
        
        # Compute decoders using training data
        # Generate activities for the training SSPs
        activities = np.array([input_ens.get_activities(ssp) for ssp in ssps]).T
        
        # Solve for decoders
        from ..encoding.decoders import DecoderSolver
        solver = DecoderSolver()
        input_ens.decoders = solver.solve(activities, positions.T, noise_sigma=0.1)
        
        # Create connection
        conn = Connection(input_ens, output_ens)
        
        net.add_ensemble(input_ens)
        net.add_ensemble(output_ens)
        net.add_connection(conn)
        
        return net, input_ens, output_ens


class NeuralSpatialMemory:
    """
    Neural implementation of spatial memory.
    
    This represents the memory M = Σ(OBJ_i ⊛ S_i) using a neural ensemble.
    """
    
    def __init__(self, ssp_generator, dimensions=512, n_neurons_per_dim=50, seed=None):
        """
        Initialize neural spatial memory
        """
        self.ssp = ssp_generator
        self.d = dimensions
        self.n_neurons_per_dim = n_neurons_per_dim
        self.seed = seed
        
        # Create ensemble to represent memory
        self.memory_ensemble = Ensemble(
            n_neurons=dimensions * n_neurons_per_dim,
            dimensions=dimensions,
            seed=seed
        )
        
        # Track vocabulary
        self.vocabulary = {}
    
    def create_query_network(self, query_type='object'):
        """
        Create a network for querying the memory
        """
        net = Network(dt=0.001)
        
        if query_type == 'object':
            # Query: M ⊛ OBJ^(-1) -> SSP representing location
            # We need:
            # 1. Memory ensemble
            # 2. Query ensemble (for OBJ^(-1))
            # 3. Convolution to get result SSP
            # 4. Decoder to get (x, y)
            
            query_ens = Ensemble(
                n_neurons=self.d * self.n_neurons_per_dim,
                dimensions=self.d,
                seed=self.seed + 1 if self.seed else None
            )
            
            result_ens = Ensemble(
                n_neurons=self.d * self.n_neurons_per_dim,
                dimensions=self.d,
                seed=self.seed + 2 if self.seed else None
            )
            
            # This would use neural convolution (simplified here)
            net.add_ensemble(self.memory_ensemble)
            net.add_ensemble(query_ens)
            net.add_ensemble(result_ens)
            
        elif query_type == 'location':
            # Query: M ⊛ S(x,y)^(-1) -> OBJ
            # Similar structure but outputs object identity
            pass
        
        return net


class NeuralSSPTransformer:
    """
    Neural implementation of spatial transformations.
    
    Can shift positions: S(x,y) ⊛ S(Δx, Δy) = S(x+Δx, y+Δy)
    """
    
    def __init__(self, ssp_generator, n_neurons_per_dim=50, seed=None):
        """
        Initialize transformer
        """
        self.ssp = ssp_generator
        self.d = ssp_generator.d
        self.n_neurons_per_dim = n_neurons_per_dim
        self.seed = seed
    
    def create_shift_network(self, delta_x, delta_y):
        """
        Create network that shifts all positions by (delta_x, delta_y).
        
        This is done by convolving the memory with S(Δx, Δy).
        """
        # Encode the shift
        shift_ssp = self.ssp.encode_position(delta_x, delta_y)
        
        net = Network(dt=0.001)
        
        # Input ensemble (current memory)
        input_ens = Ensemble(
            n_neurons=self.d * self.n_neurons_per_dim,
            dimensions=self.d,
            seed=self.seed
        )
        
        # Output ensemble (shifted memory)
        output_ens = Ensemble(
            n_neurons=self.d * self.n_neurons_per_dim,
            dimensions=self.d,
            seed=self.seed + 1 if self.seed else None
        )
        
        # Connection that convolves with shift
        def shift_function(memory):
            return self.ssp.circular_convolution(memory, shift_ssp)
        
        conn = Connection(input_ens, output_ens, function=shift_function)
        
        net.add_ensemble(input_ens)
        net.add_ensemble(output_ens)
        net.add_connection(conn)
        
        return net, input_ens, output_ens


# Utility functions

def create_neural_ssp_system(dimensions=512, n_neurons_per_dim=50, seed=42):
    """
    Create a complete neural SSP system with encoder, decoder, and memory
    """
    # Create SSP generator
    ssp = SpatialSemanticPointer(dimensions=dimensions, seed=seed)
    
    # Create neural components
    encoder = NeuralSSPEncoder(ssp, n_neurons_per_dim=n_neurons_per_dim, seed=seed)
    decoder = NeuralSSPDecoder(ssp, n_neurons_per_dim=n_neurons_per_dim, seed=seed+1)
    memory = NeuralSpatialMemory(ssp, dimensions=dimensions, 
                                  n_neurons_per_dim=n_neurons_per_dim, seed=seed+2)
    convolution = NeuralCircularConvolution(dimensions, n_neurons_per_dim=n_neurons_per_dim, 
                                           seed=seed+3)
    transformer = NeuralSSPTransformer(ssp, n_neurons_per_dim=n_neurons_per_dim, seed=seed+4)
    
    return {
        'ssp': ssp,
        'encoder': encoder,
        'decoder': decoder,
        'memory': memory,
        'convolution': convolution,
        'transformer': transformer
    }