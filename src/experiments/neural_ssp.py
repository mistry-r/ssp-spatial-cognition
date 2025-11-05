import numpy as np
import matplotlib.pyplot as plt
from ..neurons.population import Ensemble
from ..network.network import Network
from ..network.connection import Connection

def test_neural_communication_channel():
    """
    Test basic communication channel between two ensembles.
    """
    print("="*60)
    print("NEURAL COMMUNICATION CHANNEL TEST")
    print("="*60)
    
    # Create ensembles
    ens_a = Ensemble(n_neurons=500, dimensions=1, seed=42)
    ens_b = Ensemble(n_neurons=500, dimensions=1, seed=43)
    ens_b.compute_decoders(n_samples=1000, noise_sigma=0.05)
    
    # Create network
    net = Network(dt=0.001)
    net.add_ensemble(ens_a)
    net.add_ensemble(ens_b)
    
    # Connect A -> B
    conn = Connection(ens_a, ens_b)
    net.add_connection(conn)

    ens_a.compute_decoders()
    
    # Add probes
    net.probe(ens_a, 'ens_a', 'decoded')
    net.probe(ens_b, 'ens_b', 'decoded')
    
    # Set input to ensemble A
    input_value = 0.5
    ens_a.set_input(np.array([input_value]))
    
    # Run simulation
    print(f"Running simulation with input = {input_value}...")
    net.run(duration=1.0)
    
    # Get results
    data_a = net.get_probe_data('ens_a')
    data_b = net.get_probe_data('ens_b')
    times = np.arange(len(data_a)) * net.dt
    
    # Plot
    plt.figure(figsize=(12, 4))
    plt.plot(times, data_a, label='Ensemble A (input)', alpha=0.7)
    plt.plot(times, data_b, label='Ensemble B (output)', alpha=0.7)
    plt.axhline(input_value, color='k', linestyle='--', label='Target', alpha=0.3)
    plt.xlabel('Time (s)')
    plt.ylabel('Decoded value')
    plt.title('Neural Communication Channel')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/figures/neural_communication.png', dpi=150)
    print("Saved: results/figures/neural_communication.png")
    
    # Compute error
    final_output = data_b[-100:].mean()
    error = abs(final_output - input_value)
    print(f"\nTarget: {input_value:.3f}")
    print(f"Final output: {final_output:.3f}")
    print(f"Error: {error:.3f}")
    print()

def test_neural_function_computation():
    """
    Test computing a function (squaring) in a connection.
    """
    print("="*60)
    print("NEURAL FUNCTION COMPUTATION TEST")
    print("="*60)
    
    # Create ensembles
    ens_a = Ensemble(n_neurons=300, dimensions=1, seed=42)
    ens_b = Ensemble(n_neurons=300, dimensions=1, seed=43)
    ens_b.compute_decoders(n_samples=1000, noise_sigma=0.05)
    
    # Create network
    net = Network(dt=0.001)
    net.add_ensemble(ens_a)
    net.add_ensemble(ens_b)
    
    # Connect A -> B with squaring function
    conn = Connection(ens_a, ens_b, function=lambda x: x**2)
    net.add_connection(conn)

    ens_a.compute_decoders()
    
    # Add probes
    net.probe(ens_a, 'ens_a', 'decoded')
    net.probe(ens_b, 'ens_b', 'decoded')
    
    # Test different input values
    test_values = [-0.8, -0.4, 0.0, 0.4, 0.8]
    results = []
    
    for val in test_values:
        # Reset network
        for ens in net.ensembles:
            ens.reset()
        net.time = 0.0
        net.probes = {}
        net.probe(ens_a, 'ens_a', 'decoded')
        net.probe(ens_b, 'ens_b', 'decoded')
        
        # Set input
        ens_a.set_input(np.array([val]))
        
        # Run
        net.run(duration=1.0) # 0.5
        
        # Get final output
        data_b = net.get_probe_data('ens_b')
        final_output = data_b[-100:].mean()
        target = val ** 2
        
        results.append((val, final_output, target))
        print(f"Input: {val:5.2f}, Output: {final_output:5.3f}, Target: {target:5.3f}, Error: {abs(final_output - target):.3f}")
    
    # Plot
    inputs = [r[0] for r in results]
    outputs = [r[1] for r in results]
    targets = [r[2] for r in results]
    
    plt.figure(figsize=(8, 6))
    plt.scatter(inputs, outputs, label='Neural output', s=100, alpha=0.7)
    plt.plot(inputs, targets, 'r--', label='Target (x²)', linewidth=2)
    plt.xlabel('Input (x)')
    plt.ylabel('Output')
    plt.title('Neural Function Computation: f(x) = x²')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/figures/neural_function.png', dpi=150)
    print("\nSaved: results/figures/neural_function.png")
    print()

def test_neural_multiplication():
    """
    Test computing multiplication with a 2D ensemble.
    """
    print("="*60)
    print("NEURAL MULTIPLICATION TEST")
    print("="*60)
    
    # Create 2D ensemble for representing (x, y)
    ens_product = Ensemble(n_neurons=600, dimensions=2, seed=42)
    ens_result = Ensemble(n_neurons=300, dimensions=1, seed=43)
    ens_result.compute_decoders(n_samples=1000, noise_sigma=0.05)
    
    # Create network
    net = Network(dt=0.001)
    net.add_ensemble(ens_product)
    net.add_ensemble(ens_result)
    
    # Connect with multiplication function
    conn = Connection(ens_product, ens_result, 
                     function=lambda x: np.array([x[0] * x[1]]))
    net.add_connection(conn)
    
    # Add probes
    net.probe(ens_product, 'input', 'decoded')
    net.probe(ens_result, 'output', 'decoded')
    
    # Test different input pairs
    test_pairs = [
        (0.5, 0.5),
        (0.8, 0.3),
        (-0.5, 0.6),
        (-0.7, -0.4)
    ]
    
    results = []
    
    for x, y in test_pairs:
        # Reset
        for ens in net.ensembles:
            ens.reset()
        net.time = 0.0
        net.probes = {}
        net.probe(ens_product, 'input', 'decoded')
        net.probe(ens_result, 'output', 'decoded')
        
        # Set input
        ens_product.set_input(np.array([x, y]))
        
        # Run
        net.run(duration=1.0) # 0.5
        
        # Get output
        data_out = net.get_probe_data('output')
        final_output = data_out[-100:].mean()
        target = x * y
        
        results.append((x, y, final_output, target))
        print(f"Input: ({x:5.2f}, {y:5.2f}), Output: {final_output:6.3f}, Target: {target:6.3f}, Error: {abs(final_output - target):.3f}")
    
    print()

if __name__ == '__main__':
    test_neural_communication_channel()
    print()
    test_neural_function_computation()
    print()
    test_neural_multiplication()