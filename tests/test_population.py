import numpy as np
import pytest
from src.neurons.population import Ensemble

def test_ensemble_creation():
    """
    Test ensemble creation
    """
    ens = Ensemble(n_neurons=100, dimensions=2, seed=42)
    
    assert ens.n_neurons == 100
    assert ens.dimensions == 2
    assert ens.encoders.shape == (100, 2)
    assert ens.gain.shape == (100,)
    assert ens.bias.shape == (100,)

def test_ensemble_activities():
    """
    Test getting activities from ensemble
    """
    ens = Ensemble(n_neurons=50, dimensions=1, seed=42)
    
    # Test at origin
    activities = ens.get_activities(np.array([0.0]))
    assert activities.shape == (50,)
    assert np.all(activities >= 0)  # Rates should be non-negative
    
    # Test at different values
    activities1 = ens.get_activities(np.array([0.5]))
    activities2 = ens.get_activities(np.array([-0.5]))
    
    # Should get different activities
    assert not np.allclose(activities1, activities2)

def test_ensemble_decoder_computation():
    """
    Test decoder computation
    """
    ens = Ensemble(n_neurons=100, dimensions=1, seed=42)
    
    # Compute decoders
    decoders = ens.compute_decoders(n_samples=200)
    
    assert decoders.shape == (100, 1)
    
    # Test decoding
    x_test = np.array([0.5])
    activities = ens.get_activities(x_test)
    x_decoded = decoders.T @ activities
    
    # Should be reasonably close
    assert np.abs(x_decoded[0] - x_test[0]) < 0.2

def test_ensemble_2d():
    """
    Test 2D ensemble
    """
    ens = Ensemble(n_neurons=200, dimensions=2, seed=42)
    
    # Compute decoders
    ens.compute_decoders(n_samples=500)
    
    # Test decoding
    x_test = np.array([0.3, -0.5])
    activities = ens.get_activities(x_test)
    x_decoded = ens.decode(activities)
    
    assert x_decoded.shape == (2,)
    error = np.linalg.norm(x_decoded - x_test)
    assert error < 0.3

def test_ensemble_function_decoder():
    """
    Test function decoder computation.
    """
    ens = Ensemble(n_neurons=150, dimensions=1, seed=42)
    
    # Compute decoder for squaring function
    ens.compute_decoders(function=lambda x: x**2)
    
    # Test
    x_test = np.array([0.7])
    activities = ens.get_activities(x_test)
    y_decoded = ens.decode(activities)
    
    target = x_test[0] ** 2
    error = np.abs(y_decoded[0] - target)
    
    # Should approximate x^2
    assert error < 0.2

def test_ensemble_reset():
    """
    Test ensemble reset
    """
    ens = Ensemble(n_neurons=50, dimensions=1, seed=42)
    
    # Set some input
    ens.set_input(np.array([0.5]))
    assert not np.all(ens.input_current == 0)
    
    # Reset
    ens.reset()
    assert np.all(ens.input_current == 0)
    assert np.all(ens.neurons.v == 0)
    