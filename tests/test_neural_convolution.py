import numpy as np
import sys
sys.path.append('..')

from src.ssp.fractional_binding import SpatialSemanticPointer
from src.ssp.neural_convolution import (
    NeuralCircularConvolution,
    NeuralSSPEncoder,
    NeuralSSPDecoder
)

def test_neural_convolution_basic():
    """Test that neural convolution produces similar results to mathematical."""
    print("Testing neural circular convolution...")
    
    d = 256  # Smaller for faster testing
    ssp = SpatialSemanticPointer(dimensions=d, seed=42)
    conv = NeuralCircularConvolution(dimensions=d, n_neurons_per_dim=30, seed=42)
    
    # Random vectors
    a = np.random.randn(d)
    a = a / np.linalg.norm(a)
    b = np.random.randn(d)
    b = b / np.linalg.norm(b)
    
    # Mathematical convolution
    math_result = ssp.circular_convolution(a, b)
    math_result = math_result / np.linalg.norm(math_result)
    
    # Neural convolution
    neural_result = conv.convolve(a, b, duration=0.5)
    neural_result = neural_result / np.linalg.norm(neural_result)
    
    # Similarity should be high
    similarity = np.dot(math_result, neural_result)
    print(f"Similarity: {similarity:.3f}")
    
    assert similarity > 0.7, f"Similarity too low: {similarity}"
    print("✓ Neural convolution test passed\n")


def test_neural_encoder():
    """Test neural SSP encoder."""
    print("Testing neural SSP encoder...")
    
    d = 256
    ssp = SpatialSemanticPointer(dimensions=d, seed=42)
    encoder = NeuralSSPEncoder(ssp, dimensions=d, n_neurons_per_dim=30, seed=42)
    
    # Encode position
    x, y = 2.0, -1.5
    neural_ssp = encoder.encode(x, y, duration=0.5)
    
    # Mathematical version
    math_ssp = ssp.encode_position(x, y)
    
    # Similarity
    similarity = ssp.similarity(neural_ssp, math_ssp)
    print(f"Similarity: {similarity:.3f}")
    
    assert similarity > 0.7, f"Similarity too low: {similarity}"
    print("✓ Neural encoder test passed\n")


def test_neural_decoder():
    """Test neural SSP decoder."""
    print("Testing neural SSP decoder...")
    
    d = 256
    ssp = SpatialSemanticPointer(dimensions=d, seed=42)
    decoder = NeuralSSPDecoder(ssp, dimensions=d, n_neurons_per_dim=30, seed=42)
    
    # Encode position mathematically
    true_x, true_y = 1.5, -2.0
    ssp_vec = ssp.encode_position(true_x, true_y)
    
    # Neural decoding
    decoded_x, decoded_y = decoder.decode(ssp_vec, duration=0.5)
    
    # Error
    error = np.sqrt((decoded_x - true_x)**2 + (decoded_y - true_y)**2)
    print(f"True: ({true_x:.2f}, {true_y:.2f})")
    print(f"Decoded: ({decoded_x:.2f}, {decoded_y:.2f})")
    print(f"Error: {error:.3f}")
    
    assert error < 1.0, f"Error too large: {error}"
    print("✓ Neural decoder test passed\n")


def test_encode_decode_roundtrip():
    """Test encoding then decoding."""
    print("Testing encode-decode roundtrip...")
    
    d = 256
    ssp = SpatialSemanticPointer(dimensions=d, seed=42)
    encoder = NeuralSSPEncoder(ssp, dimensions=d, n_neurons_per_dim=30, seed=42)
    decoder = NeuralSSPDecoder(ssp, dimensions=d, n_neurons_per_dim=30, seed=42)
    
    # Original position
    true_x, true_y = -1.0, 2.5
    
    # Encode
    ssp_vec = encoder.encode(true_x, true_y, duration=0.5)
    
    # Decode
    decoded_x, decoded_y = decoder.decode(ssp_vec, duration=0.5)
    
    # Error
    error = np.sqrt((decoded_x - true_x)**2 + (decoded_y - true_y)**2)
    print(f"True: ({true_x:.2f}, {true_y:.2f})")
    print(f"Decoded: ({decoded_x:.2f}, {decoded_y:.2f})")
    print(f"Error: {error:.3f}")
    
    assert error < 1.5, f"Roundtrip error too large: {error}"
    print("✓ Encode-decode roundtrip test passed\n")


if __name__ == '__main__':
    print("=" * 60)
    print("NEURAL CONVOLUTION TESTS")
    print("=" * 60 + "\n")
    
    test_neural_convolution_basic()
    test_neural_encoder()
    test_neural_decoder()
    test_encode_decode_roundtrip()
    
    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)