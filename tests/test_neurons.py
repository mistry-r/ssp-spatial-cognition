import numpy as np
import pytest
from src.neurons.lif import LIFNeuron, LIFPopulation

def test_lif_neuron_basic():
    """
    Test basic LIF neuron functionality
    """
    neuron = LIFNeuron(tau_rc=0.02, tau_ref=0.002)
    
    # Should not spike with zero input
    spike = neuron.step(J=0.0, dt=0.001)
    assert not spike
    
    # Reset
    neuron.reset()
    assert neuron.v == 0.0

def test_lif_rate_approximation():
    """
    Test LIF rate approximation
    """
    neuron = LIFNeuron(tau_rc=0.02, tau_ref=0.002)
    
    # Below threshold should give zero rate
    rate = neuron.rate_approximation(J=0.5)
    assert rate == 0.0
    
    # Above threshold should give positive rate
    rate = neuron.rate_approximation(J=2.0)
    assert rate > 0.0
    
    # Higher current should give higher rate
    rate1 = neuron.rate_approximation(J=2.0)
    rate2 = neuron.rate_approximation(J=3.0)
    assert rate2 > rate1

def test_lif_spiking():
    """
    Test that LIF neuron actually spikes
    """
    neuron = LIFNeuron(tau_rc=0.02, tau_ref=0.002)
    
    # Apply strong current
    J = 5.0
    dt = 0.001
    
    spikes = []
    for _ in range(100):
        spike = neuron.step(J, dt)
        spikes.append(spike)
    
    # Should have spiked at least once
    assert sum(spikes) > 0

def test_lif_population():
    """
    Test LIF population
    """
    n_neurons = 10
    pop = LIFPopulation(n_neurons=n_neurons)
    
    # Apply different currents to each neuron
    J = np.linspace(0, 5, n_neurons)
    dt = 0.001
    
    # Run for some time
    spike_counts = np.zeros(n_neurons)
    for _ in range(1000):
        spikes = pop.step(J, dt)
        spike_counts += spikes
    
    # Higher current neurons should spike more
    assert spike_counts[-1] > spike_counts[0]

def test_lif_refractory_period():
    """
    Test refractory period prevents immediate re-spiking
    """
    neuron = LIFNeuron(tau_rc=0.02, tau_ref=0.002)
    
    J = 10.0  # Very strong current
    dt = 0.0001
    
    # Wait for first spike
    spiked = False
    for _ in range(1000):
        if neuron.step(J, dt):
            spiked = True
            break
    
    assert spiked
    
    # Should not spike immediately after
    spike = neuron.step(J, dt)
    assert not spike

def test_rate_approximation_vectorized():
    """
    Test vectorized rate approximation
    """
    neuron = LIFNeuron()
    
    J = np.array([0.5, 1.5, 2.5, 3.5])
    rates = neuron.rate_approximation(J)
    
    assert len(rates) == len(J)
    assert rates[0] == 0.0  # Below threshold
    assert rates[1] > 0.0   # Above threshold
    assert rates[3] > rates[2]  # Higher current = higher rate