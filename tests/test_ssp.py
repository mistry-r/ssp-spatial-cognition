import pytest
import numpy as np
from src.ssp.fractional_binding import SpatialSemanticPointer

class TestSpatialSemanticPointer:
    
    def test_initialization(self):
        """Test that SSP initializes correctly."""
        ssp = SpatialSemanticPointer(dimensions=256)
        assert ssp.d == 256
        assert len(ssp.X) == 256
        assert len(ssp.Y) == 256
    
    def test_dimensions_must_be_even(self):
        """Test that odd dimensions raise an error."""
        with pytest.raises(ValueError):
            ssp = SpatialSemanticPointer(dimensions=255)
    
    def test_unitary_vector_unit_length(self):
        """Test that generated unitary vectors have unit length."""
        ssp = SpatialSemanticPointer(dimensions=512, seed=42)
        
        # X and Y should be unit vectors
        assert np.isclose(np.linalg.norm(ssp.X), 1.0, atol=1e-6)
        assert np.isclose(np.linalg.norm(ssp.Y), 1.0, atol=1e-6)
    
    def test_unitary_property(self):
        """Test that vectors maintain length under binding."""
        ssp = SpatialSemanticPointer(dimensions=512, seed=42)
        
        # X^2 should have same length as X
        X_squared = ssp.circular_convolution(ssp.X, ssp.X)
        assert np.isclose(np.linalg.norm(X_squared), np.linalg.norm(ssp.X), atol=1e-3)
        
        # X^3 should have same length as X
        X_cubed = ssp.circular_convolution(X_squared, ssp.X)
        assert np.isclose(np.linalg.norm(X_cubed), np.linalg.norm(ssp.X), atol=1e-3)
    
    def test_circular_convolution_commutative(self):
        """Test that circular convolution is commutative."""
        ssp = SpatialSemanticPointer(dimensions=256, seed=42)
        
        a = ssp.X
        b = ssp.Y
        
        ab = ssp.circular_convolution(a, b)
        ba = ssp.circular_convolution(b, a)
        
        assert np.allclose(ab, ba, atol=1e-10)
    
    def test_fractional_binding_property(self):
        """Test Equation 3: X^a ⊛ X^b = X^{a+b}."""
        ssp = SpatialSemanticPointer(dimensions=512, seed=42)
        
        a, b = 2.5, 1.5
        
        # Compute X^a ⊛ X^b
        X_a = ssp.fractional_power(ssp.X, a)
        X_b = ssp.fractional_power(ssp.X, b)
        X_a_conv_b = ssp.circular_convolution(X_a, X_b)
        
        # Compute X^{a+b}
        X_a_plus_b = ssp.fractional_power(ssp.X, a + b)
        
        # Should be approximately equal
        similarity = ssp.similarity(X_a_conv_b, X_a_plus_b)
        assert similarity > 0.99
    
    def test_inverse_property(self):
        """Test that v ⊛ v^{-1} ≈ identity."""
        ssp = SpatialSemanticPointer(dimensions=512, seed=42)
        
        # Get inverse
        X_inv = ssp.get_inverse(ssp.X)
        
        # Convolve with original
        result = ssp.circular_convolution(ssp.X, X_inv)
        
        # Should be close to identity (which has high autocorrelation at lag 0)
        # The identity in circular convolution is [1, 0, 0, ..., 0]
        identity = np.zeros(ssp.d)
        identity[0] = 1.0
        
        # Result should have high value at first element
        assert result[0] > 0.9
        assert np.abs(result[1:]).max() < 0.1
    
    def test_encode_decode_position(self):
        """Test that positions can be encoded and decoded."""
        ssp = SpatialSemanticPointer(dimensions=512, seed=42)
        
        # Test several positions
        test_positions = [
            (0, 0),
            (2.5, -3.1),
            (-4.2, 1.7),
            (3.0, 3.0)
        ]
        
        for true_x, true_y in test_positions:
            # Encode
            pos_ssp = ssp.encode_position(true_x, true_y)
            
            # Decode
            decoded_x, decoded_y = ssp.decode_position(
                pos_ssp, bounds=(-5, 5), resolution=100
            )
            
            # Check error
            error = np.sqrt((decoded_x - true_x)**2 + (decoded_y - true_y)**2)
            assert error < 0.5, f"Position ({true_x}, {true_y}) decoded as ({decoded_x}, {decoded_y})"
    
    def test_similarity_range(self):
        """Test that similarity is in valid range."""
        ssp = SpatialSemanticPointer(dimensions=256, seed=42)
        
        # Same vector should have similarity ~1
        assert np.isclose(ssp.similarity(ssp.X, ssp.X), 1.0, atol=1e-6)
        
        # Random vectors should have similarity ~0
        random_vec = np.random.randn(256)
        random_vec = random_vec / np.linalg.norm(random_vec)
        sim = ssp.similarity(ssp.X, random_vec)
        assert abs(sim) < 0.2  # Should be close to 0
    
    def test_encode_position_unit_length(self):
        """Test that encoded positions are unit vectors."""
        ssp = SpatialSemanticPointer(dimensions=512, seed=42)
        
        pos_ssp = ssp.encode_position(1.5, -2.3)
        assert np.isclose(np.linalg.norm(pos_ssp), 1.0, atol=1e-6)