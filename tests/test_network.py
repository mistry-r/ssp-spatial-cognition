import numpy as np
import pytest
from src.neurons.population import Ensemble
from src.network.network import Network
from src.network.connection import Connection

def test_network_creation():
    """Test network creation."""
    net = Network(dt=0.001)
    assert net.dt == 0.001
    assert len(net.ensembles) == 0
    assert len(net.connections) == 0

def test_network_communication_channel():
    """Test simple communication channel."""
    # Create ensembles
    ens_a = Ensemble(n_neurons=100, dimensions=1, seed=42)
    ens_b = Ensemble(n_neurons=100, dimensions=1, seed=43)
    
    # Create network
    net = Network(dt=0.001)
    net.add_ensemble(ens_a)
    net.add_ensemble(ens_b)
    
    # Connect
    conn = Connection(ens_a, ens_b)
    net.add_connection(conn)
    
    # Set input
    ens_a.set_input(np.array([0.5]))
    
    # Run
    net.run(duration=0.1)
    
    # Should have run
    assert net.time > 0

def test_network_with_probes():
    """Test network with probes."""
    ens = Ensemble(n_neurons=50, dimensions=1, seed=42)
    
    # Compute decoders before using ensemble
    ens.compute_decoders()
    
    net = Network(dt=0.001)
    net.add_ensemble(ens)
    net.probe(ens, 'test', 'decoded')
    
    # Set input and run
    ens.set_input(np.array([0.3]))
    net.run(duration=0.1)
    
    # Get data
    data = net.get_probe_data('test')
    
    assert len(data) > 0
    assert data.shape[1] == 1  # 1D ensemble

def test_network_function_computation():
    """Test function computation in network."""
    ens_a = Ensemble(n_neurons=150, dimensions=1, seed=42)
    ens_b = Ensemble(n_neurons=150, dimensions=1, seed=43)
    
    net = Network(dt=0.001)
    net.add_ensemble(ens_a)
    net.add_ensemble(ens_b)
    
    # Connect with squaring function
    conn = Connection(ens_a, ens_b, function=lambda x: x**2)
    net.add_connection(conn)
    
    # Probe
    net.probe(ens_b, 'output', 'decoded')
    
    # Compute decoders before running
    ens_a.compute_decoders()
    ens_b.compute_decoders() 
    
    # Set input
    input_val = 0.6
    ens_a.set_input(np.array([input_val]))
    
    # Run
    net.run(duration=0.3)
    
    # Check output
    data = net.get_probe_data('output')
    final_output = data[-50:].mean()
    
    target = input_val ** 2
    error = abs(final_output - target)
    
    # Should be reasonably accurate
    assert error < 0.15

def test_network_2d_ensemble():
    """Test 2D ensemble in network."""
    ens = Ensemble(n_neurons=200, dimensions=2, seed=42)
    
    # IMPORTANT: Compute decoders
    ens.compute_decoders()
    
    net = Network(dt=0.001)
    net.add_ensemble(ens)
    net.probe(ens, 'state', 'decoded')
    
    # Set 2D input
    ens.set_input(np.array([0.4, -0.3]))
    
    # Run
    net.run(duration=0.2)
    
    # Get data
    data = net.get_probe_data('state')
    
    assert data.shape[1] == 2  # 2D
    