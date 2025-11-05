import numpy as np
import pytest
from src.encoding.encoders import EncoderGenerator
from src.encoding.decoders import DecoderSolver

def test_random_encoders():
    """Test random encoder generation."""
    gen = EncoderGenerator(seed=42)
    encoders = gen.generate_encoders(100, 3, encoder_type='random')
    
    assert encoders.shape == (100, 3)
    
    # Should be unit length
    norms = np.linalg.norm(encoders, axis=1)
    assert np.allclose(norms, 1.0)

def test_grid_encoders():
    """Test grid encoder generation (2D)."""
    gen = EncoderGenerator(seed=42)
    encoders = gen.generate_encoders(8, 2, encoder_type='grid')
    
    assert encoders.shape == (8, 2)
    
    # Should be evenly spaced around circle
    angles = np.arctan2(encoders[:, 1], encoders[:, 0])
    angle_diffs = np.diff(np.sort(angles))
    
    # Should be approximately equal spacing
    assert np.std(angle_diffs) < 0.1

def test_axis_aligned_encoders():
    """Test axis-aligned encoders."""
    gen = EncoderGenerator(seed=42)
    encoders = gen.generate_encoders(6, 3, encoder_type='axis_aligned')
    
    assert encoders.shape == (6, 3)
    
    # First 6 should be axis-aligned
    # [1,0,0], [-1,0,0], [0,1,0], [0,-1,0], [0,0,1], [0,0,-1]
    for i in range(6):
        # Should have exactly one non-zero component
        non_zero = np.sum(np.abs(encoders[i]) > 0.5)
        assert non_zero == 1

def test_decoder_solver():
    """Test decoder solver."""
    solver = DecoderSolver()
    
    # Create fake activities and targets
    n_neurons = 50
    n_samples = 100
    dimensions = 2
    
    activities = np.random.rand(n_neurons, n_samples)
    targets = np.random.rand(dimensions, n_samples)
    
    # Solve
    decoders = solver.solve(activities, targets, noise_sigma=0.1)
    
    assert decoders.shape == (n_neurons, dimensions)

def test_decoder_identity():
    """Test that decoders can approximate identity."""
    solver = DecoderSolver()
    
    # Simple case: activities are just the targets with noise
    n_samples = 200
    dimensions = 2
    
    targets = np.random.uniform(-1, 1, (dimensions, n_samples))
    activities = targets + 0.1 * np.random.randn(dimensions, n_samples)
    
    decoders = solver.solve(activities, targets, noise_sigma=0.05)
    
    # Test decoding
    test_activities = targets[:, :10]
    decoded = decoders.T @ test_activities
    
    # Should be close to targets
    error = np.linalg.norm(decoded - targets[:, :10]) / 10
    assert error < 0.5

def test_function_decoder():
    """Test function decoder."""
    solver = DecoderSolver()
    
    # Create fake scenario
    n_neurons = 100
    n_samples = 300
    
    # Input samples
    x_samples = np.random.uniform(-1, 1, (n_samples, 1))
    
    # Activities (random but correlated with input)
    activities = np.abs(x_samples.T * np.random.rand(n_neurons, 1)) + 0.1
    
    # Solve for squaring function
    decoders = solver.solve_function(
        activities, x_samples, lambda x: x**2, noise_sigma=0.1
    )
    
    assert decoders.shape[0] == n_neurons
    