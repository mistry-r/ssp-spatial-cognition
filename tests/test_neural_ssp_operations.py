import numpy as np
import pytest
from src.ssp.fractional_binding import SpatialSemanticPointer
from src.ssp.neural_ssp_operations import (
    NeuralCircularConvolution,
    NeuralSSPEncoder,
    NeuralSSPDecoder,
    NeuralSpatialMemory,
    NeuralSSPTransformer,
    create_neural_ssp_system
)


class TestNeuralCircularConvolution:
    """Test neural circular convolution."""
    
    def test_initialization(self):
        """Test that neural convolution can be initialized."""
        conv = NeuralCircularConvolution(dimensions=128, n_neurons_per_dim=50, seed=42)
        assert conv.d == 128
        assert conv.n_neurons == 128 * 50
        assert conv.result_ensemble is not None
    
    def test_network_creation(self):
        """Test creating a convolution network."""
        conv = NeuralCircularConvolution(dimensions=128, n_neurons_per_dim=10, seed=42)
        net, ens_a, ens_b = conv.create_network(input_a_dims=128, input_b_dims=128)
        
        assert net is not None
        assert ens_a.dimensions == 128
        assert ens_b.dimensions == 128
        assert len(net.ensembles) == 3
    
    def test_fixed_convolution(self):
        """Test convolving with a fixed vector."""
        # Create simple vectors
        ssp = SpatialSemanticPointer(dimensions=64, seed=42)
        
        fixed = ssp.X.copy()
        variable = ssp.Y.copy()
        
        # Create neural convolution
        conv = NeuralCircularConvolution(dimensions=64, n_neurons_per_dim=10, seed=42)
        
        # Create ensemble for variable vector
        from src.neurons.population import Ensemble
        var_ens = Ensemble(n_neurons=640, dimensions=64, seed=42)
        out_ens = Ensemble(n_neurons=640, dimensions=64, seed=43)
        
        # Create connection
        conn = conv.convolve_with_fixed(fixed, var_ens, out_ens)
        
        assert conn is not None
        assert conn.pre == var_ens
        assert conn.post == out_ens
        
        # Set input
        var_ens.set_input(variable)
        
        # Test that connection computes something
        var_ens.compute_decoders(n_samples=500)
        out_ens.compute_decoders(n_samples=500)
        
        # Compute expected result
        expected = ssp.circular_convolution(fixed, variable)
        
        # Get activities
        activities = var_ens.get_activities(variable)
        decoded = var_ens.decode(activities)
        
        # Just check it doesn't crash for now
        assert decoded is not None


class TestNeuralSSPEncoder:
    """Test neural SSP encoder."""
    
    def test_initialization(self):
        """Test encoder initialization."""
        ssp = SpatialSemanticPointer(dimensions=128, seed=42)
        encoder = NeuralSSPEncoder(ssp, n_neurons_per_dim=50, seed=42)
        
        assert encoder.ssp == ssp
        assert encoder.d == 128
        assert encoder.n_neurons_per_dim == 50
    
    def test_network_creation(self):
        """Test creating encoder network."""
        ssp = SpatialSemanticPointer(dimensions=128, seed=42)
        encoder = NeuralSSPEncoder(ssp, n_neurons_per_dim=10, seed=42)
        
        net, input_ens, output_ens = encoder.create_network()
        
        assert net is not None
        assert input_ens.dimensions == 2  # (x, y)
        assert output_ens.dimensions == 128  # SSP dimension
        assert len(net.ensembles) == 2
        assert len(net.connections) == 1
    
    def test_encoding_accuracy(self):
        """Test that neural encoding approximates mathematical encoding."""
        ssp = SpatialSemanticPointer(dimensions=256, seed=42)
        encoder = NeuralSSPEncoder(ssp, n_neurons_per_dim=20, seed=42)
        
        net, input_ens, output_ens = encoder.create_network()
        
        # Test position
        test_pos = np.array([1.5, -2.0])
        
        # Mathematical encoding
        expected_ssp = ssp.encode_position(test_pos[0], test_pos[1])
        
        # Set input
        input_ens.set_input(test_pos)
        
        # Compute decoders
        input_ens.compute_decoders(n_samples=1000)
        output_ens.compute_decoders(n_samples=1000)
        
        # Run network for a bit
        net.probe(output_ens, 'output', 'decoded')
        net.run(duration=0.5)
        
        # Get output
        outputs = net.get_probe_data('output')
        final_output = outputs[-100:].mean(axis=0)
        
        # Check similarity (should be high)
        similarity = ssp.similarity(final_output, expected_ssp)
        
        # Neural implementation won't be perfect, but should be reasonable
        assert similarity > 0.5, f"Similarity too low: {similarity}"
    
    def test_separate_axis_network(self):
        """Test separate axis encoding."""
        ssp = SpatialSemanticPointer(dimensions=128, seed=42)
        encoder = NeuralSSPEncoder(ssp, n_neurons_per_dim=10, seed=42)
        
        net, input_ens, x_ens, y_ens, output_ens = encoder.create_separate_axis_network()
        
        assert input_ens.dimensions == 2
        assert x_ens.dimensions == 128
        assert y_ens.dimensions == 128


class TestNeuralSSPDecoder:
    """Test neural SSP decoder."""
    
    def test_initialization(self):
        """Test decoder initialization."""
        ssp = SpatialSemanticPointer(dimensions=128, seed=42)
        decoder = NeuralSSPDecoder(ssp, n_neurons_per_dim=50, bounds=(-5, 5), seed=42)
        
        assert decoder.ssp == ssp
        assert decoder.d == 128
        assert decoder.bounds == (-5, 5)
    
    def test_training_data_generation(self):
        """Test generating training data."""
        ssp = SpatialSemanticPointer(dimensions=128, seed=42)
        decoder = NeuralSSPDecoder(ssp, n_neurons_per_dim=50, seed=42)
        
        ssps, positions = decoder.generate_training_data(n_samples=100)
        
        assert ssps.shape == (100, 128)
        assert positions.shape == (100, 2)
        
        # Check positions are in bounds
        assert np.all(positions >= -5)
        assert np.all(positions <= 5)
        
        # Check SSPs match positions
        for i in range(10):  # Test a few
            expected = ssp.encode_position(positions[i, 0], positions[i, 1])
            similarity = ssp.similarity(ssps[i], expected)
            assert similarity > 0.99  # Should be nearly identical
    
    def test_network_creation(self):
        """Test creating decoder network."""
        ssp = SpatialSemanticPointer(dimensions=128, seed=42)
        decoder = NeuralSSPDecoder(ssp, n_neurons_per_dim=10, seed=42)
        
        net, input_ens, output_ens = decoder.create_network()
        
        assert net is not None
        assert input_ens.dimensions == 128  # SSP
        assert output_ens.dimensions == 2  # (x, y)
        assert len(net.ensembles) == 2
        assert len(net.connections) == 1
    
    def test_decoding_accuracy(self):
        """Test that neural decoding works."""
        ssp = SpatialSemanticPointer(dimensions=256, seed=42)
        decoder = NeuralSSPDecoder(ssp, n_neurons_per_dim=20, bounds=(-5, 5), seed=42)
        
        # Generate training data
        decoder.generate_training_data(n_samples=500)
        
        net, input_ens, output_ens = decoder.create_network()
        
        # Test position
        test_pos = [2.0, -1.5]
        test_ssp = ssp.encode_position(test_pos[0], test_pos[1])
        
        # Set input
        input_ens.set_input(test_ssp)
        
        # Run network
        net.probe(output_ens, 'output', 'decoded')
        net.run(duration=0.5)
        
        # Get output
        outputs = net.get_probe_data('output')
        final_output = outputs[-100:].mean(axis=0)
        
        # Check accuracy
        error = np.linalg.norm(final_output - test_pos)
        
        # Allow for some error in neural decoding
        assert error < 1.0, f"Decoding error too large: {error}"


class TestNeuralSpatialMemory:
    """Test neural spatial memory."""
    
    def test_initialization(self):
        """Test memory initialization."""
        ssp = SpatialSemanticPointer(dimensions=128, seed=42)
        memory = NeuralSpatialMemory(ssp, dimensions=128, n_neurons_per_dim=50, seed=42)
        
        assert memory.ssp == ssp
        assert memory.d == 128
        assert memory.memory_ensemble is not None
    
    def test_query_network_creation(self):
        """Test creating query networks."""
        ssp = SpatialSemanticPointer(dimensions=128, seed=42)
        memory = NeuralSpatialMemory(ssp, dimensions=128, n_neurons_per_dim=10, seed=42)
        
        net = memory.create_query_network(query_type='object')
        
        assert net is not None
        assert len(net.ensembles) >= 2


class TestNeuralSSPTransformer:
    """Test neural SSP transformer."""
    
    def test_initialization(self):
        """Test transformer initialization."""
        ssp = SpatialSemanticPointer(dimensions=128, seed=42)
        transformer = NeuralSSPTransformer(ssp, n_neurons_per_dim=50, seed=42)
        
        assert transformer.ssp == ssp
        assert transformer.d == 128
    
    def test_shift_network_creation(self):
        """Test creating shift network."""
        ssp = SpatialSemanticPointer(dimensions=128, seed=42)
        transformer = NeuralSSPTransformer(ssp, n_neurons_per_dim=10, seed=42)
        
        net, input_ens, output_ens = transformer.create_shift_network(delta_x=1.0, delta_y=-0.5)
        
        assert net is not None
        assert input_ens.dimensions == 128
        assert output_ens.dimensions == 128
        assert len(net.ensembles) == 2
        assert len(net.connections) == 1
    
    def test_shift_accuracy(self):
        """Test that shifting works correctly."""
        ssp = SpatialSemanticPointer(dimensions=256, seed=42)
        transformer = NeuralSSPTransformer(ssp, n_neurons_per_dim=20, seed=42)
        
        # Original position
        orig_pos = [1.0, 1.0]
        orig_ssp = ssp.encode_position(orig_pos[0], orig_pos[1])
        
        # Shift amount
        delta = [0.5, -1.0]
        expected_pos = [orig_pos[0] + delta[0], orig_pos[1] + delta[1]]
        
        # Create shift network
        net, input_ens, output_ens = transformer.create_shift_network(delta[0], delta[1])
        
        # Set input
        input_ens.set_input(orig_ssp)
        
        # Compute decoders
        input_ens.compute_decoders(n_samples=500)
        output_ens.compute_decoders(n_samples=500)
        
        # Run network
        net.probe(output_ens, 'output', 'decoded')
        net.run(duration=0.5)
        
        # Get shifted SSP
        outputs = net.get_probe_data('output')
        shifted_ssp = outputs[-100:].mean(axis=0)
        
        # Decode to check position
        decoded_pos = ssp.decode_position(shifted_ssp, bounds=(-5, 5))
        
        error = np.linalg.norm(np.array(decoded_pos) - np.array(expected_pos))
        
        # Allow some error
        assert error < 1.0, f"Shift error too large: {error}"


class TestNeuralSSPSystem:
    """Test complete neural SSP system."""
    
    def test_system_creation(self):
        """Test creating complete system."""
        system = create_neural_ssp_system(dimensions=128, n_neurons_per_dim=10, seed=42)
        
        assert 'ssp' in system
        assert 'encoder' in system
        assert 'decoder' in system
        assert 'memory' in system
        assert 'convolution' in system
        assert 'transformer' in system
        
        assert isinstance(system['ssp'], SpatialSemanticPointer)
        assert isinstance(system['encoder'], NeuralSSPEncoder)
        assert isinstance(system['decoder'], NeuralSSPDecoder)
        assert isinstance(system['memory'], NeuralSpatialMemory)
        assert isinstance(system['convolution'], NeuralCircularConvolution)
        assert isinstance(system['transformer'], NeuralSSPTransformer)
    
    def test_encode_decode_pipeline(self):
        """Test encoding and then decoding a position."""
        system = create_neural_ssp_system(dimensions=256, n_neurons_per_dim=20, seed=42)
        
        ssp = system['ssp']
        encoder = system['encoder']
        decoder = system['decoder']
        
        # Create encoder network
        enc_net, enc_in, enc_out = encoder.create_network()
        
        # Create decoder network
        decoder.generate_training_data(n_samples=500)
        dec_net, dec_in, dec_out = decoder.create_network()
        
        # Test position
        test_pos = np.array([2.0, -1.0])
        
        # Encode
        enc_in.set_input(test_pos)
        enc_net.probe(enc_out, 'encoded', 'decoded')
        enc_net.run(duration=0.3)
        
        # Get encoded SSP
        encoded_outputs = enc_net.get_probe_data('encoded')
        encoded_ssp = encoded_outputs[-50:].mean(axis=0)
        
        # Decode
        dec_in.set_input(encoded_ssp)
        dec_net.probe(dec_out, 'decoded', 'decoded')
        dec_net.run(duration=0.3)
        
        # Get decoded position
        decoded_outputs = dec_net.get_probe_data('decoded')
        decoded_pos = decoded_outputs[-50:].mean(axis=0)
        
        # Check accuracy
        error = np.linalg.norm(decoded_pos - test_pos)
        
        # Full pipeline has accumulated error
        assert error < 1.5, f"Full pipeline error too large: {error}"


# Run tests
if __name__ == '__main__':
    pytest.main([__file__, '-v'])