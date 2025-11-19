import numpy as np
import matplotlib.pyplot as plt
from ..ssp.fractional_binding import SpatialSemanticPointer
from ..ssp.spatial_memory import SpatialMemory
from ..ssp.neural_ssp_operations import (
    NeuralSSPEncoder,
    NeuralSSPDecoder,
    NeuralSSPTransformer,
    create_neural_ssp_system
)

def experiment_1_neural_encoding():
    """
    Experiment 1: Test neural encoding of positions.
    
    Compare mathematical vs neural encoding accuracy.
    """
    print("=" * 70)
    print("EXPERIMENT 1: Neural Position Encoding")
    print("=" * 70)
    print()
    
    # Create SSP system
    ssp = SpatialSemanticPointer(dimensions=512, seed=42)
    encoder = NeuralSSPEncoder(ssp, n_neurons_per_dim=50, seed=42)
    
    # Create network
    net, input_ens, output_ens = encoder.create_network()
    
    # Compute decoders
    print("Computing decoders...")
    input_ens.compute_decoders(n_samples=1000)
    output_ens.compute_decoders(n_samples=1000)
    print("Done!")
    print()
    
    # Test multiple positions
    test_positions = [
        (0.0, 0.0),
        (1.0, 1.0),
        (2.0, -1.5),
        (-3.0, 2.5),
        (4.0, -4.0)
    ]
    
    results = []
    
    print("Testing positions:")
    print("-" * 70)
    
    for x, y in test_positions:
        # Mathematical encoding
        math_ssp = ssp.encode_position(x, y)
        
        # Neural encoding
        input_ens.set_input(np.array([x, y]))
        net.probe(output_ens, 'output', 'decoded')
        net.run(duration=0.5)
        
        outputs = net.get_probe_data('output')
        neural_ssp = outputs[-100:].mean(axis=0)
        
        # Compare
        similarity = ssp.similarity(math_ssp, neural_ssp)
        
        # Decode both to check position accuracy
        math_decoded = ssp.decode_position(math_ssp, bounds=(-5, 5))
        neural_decoded = ssp.decode_position(neural_ssp, bounds=(-5, 5))
        
        position_error = np.linalg.norm(
            np.array(math_decoded) - np.array([x, y])
        )
        neural_error = np.linalg.norm(
            np.array(neural_decoded) - np.array([x, y])
        )
        
        print(f"Position: ({x:5.1f}, {y:5.1f})")
        print(f"  Math decoded:   ({math_decoded[0]:5.2f}, {math_decoded[1]:5.2f}), error: {position_error:.3f}")
        print(f"  Neural decoded: ({neural_decoded[0]:5.2f}, {neural_decoded[1]:5.2f}), error: {neural_error:.3f}")
        print(f"  Similarity: {similarity:.3f}")
        print()
        
        results.append({
            'position': (x, y),
            'similarity': similarity,
            'math_error': position_error,
            'neural_error': neural_error
        })
        
        # Reset network for next test
        net.time = 0.0
        net.probes = {}
        for ens in net.ensembles:
            ens.reset()
    
    # Summary
    avg_similarity = np.mean([r['similarity'] for r in results])
    avg_neural_error = np.mean([r['neural_error'] for r in results])
    
    print("=" * 70)
    print(f"Average similarity: {avg_similarity:.3f}")
    print(f"Average neural error: {avg_neural_error:.3f}")
    print("=" * 70)
    print()
    
    return results


def experiment_2_neural_decoding():
    """
    Experiment 2: Test neural decoding of SSPs to positions.
    """
    print("=" * 70)
    print("EXPERIMENT 2: Neural Position Decoding")
    print("=" * 70)
    print()
    
    # Create system
    ssp = SpatialSemanticPointer(dimensions=512, seed=42)
    decoder = NeuralSSPDecoder(ssp, n_neurons_per_dim=50, bounds=(-5, 5), seed=42)
    
    # Generate training data
    print("Generating training data...")
    decoder.generate_training_data(n_samples=1000)
    print("Done!")
    print()
    
    # Create network
    net, input_ens, output_ens = decoder.create_network()
    
    # Test positions
    test_positions = [
        (0.0, 0.0),
        (1.5, -2.0),
        (-3.5, 1.0),
        (2.0, 3.0),
        (-1.0, -1.0)
    ]
    
    results = []
    
    print("Testing decoding:")
    print("-" * 70)
    
    for x, y in test_positions:
        # Encode position
        test_ssp = ssp.encode_position(x, y)
        
        # Mathematical decoding
        math_decoded = ssp.decode_position(test_ssp, bounds=(-5, 5))
        
        # Neural decoding
        input_ens.set_input(test_ssp)
        net.probe(output_ens, 'output', 'decoded')
        net.run(duration=0.5)
        
        outputs = net.get_probe_data('output')
        neural_decoded = outputs[-100:].mean(axis=0)
        
        # Errors
        math_error = np.linalg.norm(np.array(math_decoded) - np.array([x, y]))
        neural_error = np.linalg.norm(neural_decoded - np.array([x, y]))
        
        print(f"True position:   ({x:5.1f}, {y:5.1f})")
        print(f"  Math decoded:  ({math_decoded[0]:5.2f}, {math_decoded[1]:5.2f}), error: {math_error:.3f}")
        print(f"  Neural decoded: ({neural_decoded[0]:5.2f}, {neural_decoded[1]:5.2f}), error: {neural_error:.3f}")
        print()
        
        results.append({
            'position': (x, y),
            'math_error': math_error,
            'neural_error': neural_error
        })
        
        # Reset
        net.time = 0.0
        net.probes = {}
        for ens in net.ensembles:
            ens.reset()
    
    # Summary
    avg_math_error = np.mean([r['math_error'] for r in results])
    avg_neural_error = np.mean([r['neural_error'] for r in results])
    
    print("=" * 70)
    print(f"Average math decoding error:   {avg_math_error:.3f}")
    print(f"Average neural decoding error: {avg_neural_error:.3f}")
    print("=" * 70)
    print()
    
    return results


def experiment_3_neural_transformation():
    """
    Experiment 3: Test neural spatial transformations (shifting).
    """
    print("=" * 70)
    print("EXPERIMENT 3: Neural Spatial Transformations")
    print("=" * 70)
    print()
    
    # Create system
    ssp = SpatialSemanticPointer(dimensions=512, seed=42)
    transformer = NeuralSSPTransformer(ssp, n_neurons_per_dim=50, seed=42)
    
    # Test position and shift
    original_pos = [2.0, 1.0]
    shift_amount = [1.0, -2.0]
    expected_pos = [
        original_pos[0] + shift_amount[0],
        original_pos[1] + shift_amount[1]
    ]
    
    print(f"Original position: ({original_pos[0]:.1f}, {original_pos[1]:.1f})")
    print(f"Shift amount:      ({shift_amount[0]:.1f}, {shift_amount[1]:.1f})")
    print(f"Expected position: ({expected_pos[0]:.1f}, {expected_pos[1]:.1f})")
    print()
    
    # Encode original position
    orig_ssp = ssp.encode_position(original_pos[0], original_pos[1])
    
    # Mathematical transformation
    shift_ssp = ssp.encode_position(shift_amount[0], shift_amount[1])
    math_shifted = ssp.circular_convolution(orig_ssp, shift_ssp)
    math_result = ssp.decode_position(math_shifted, bounds=(-5, 5))
    
    print("Mathematical transformation:")
    print(f"  Result: ({math_result[0]:.2f}, {math_result[1]:.2f})")
    math_error = np.linalg.norm(np.array(math_result) - np.array(expected_pos))
    print(f"  Error:  {math_error:.3f}")
    print()
    
    # Neural transformation
    net, input_ens, output_ens = transformer.create_shift_network(
        shift_amount[0], shift_amount[1]
    )
    
    print("Computing decoders...")
    input_ens.compute_decoders(n_samples=1000)
    output_ens.compute_decoders(n_samples=1000)
    print("Done!")
    print()
    
    input_ens.set_input(orig_ssp)
    net.probe(output_ens, 'output', 'decoded')
    net.run(duration=0.5)
    
    outputs = net.get_probe_data('output')
    neural_shifted = outputs[-100:].mean(axis=0)
    neural_result = ssp.decode_position(neural_shifted, bounds=(-5, 5))
    
    print("Neural transformation:")
    print(f"  Result: ({neural_result[0]:.2f}, {neural_result[1]:.2f})")
    neural_error = np.linalg.norm(np.array(neural_result) - np.array(expected_pos))
    print(f"  Error:  {neural_error:.3f}")
    print()
    
    print("=" * 70)
    print(f"Mathematical error: {math_error:.3f}")
    print(f"Neural error:       {neural_error:.3f}")
    print("=" * 70)
    print()
    
    return {
        'math_error': math_error,
        'neural_error': neural_error,
        'math_result': math_result,
        'neural_result': neural_result,
        'expected': expected_pos
    }


def experiment_4_encode_decode_pipeline():
    """
    Experiment 4: Full encode-decode pipeline.
    
    Test: position -> neural encode -> SSP -> neural decode -> position
    """
    print("=" * 70)
    print("EXPERIMENT 4: Full Encode-Decode Pipeline")
    print("=" * 70)
    print()
    
    # Create system
    system = create_neural_ssp_system(dimensions=512, n_neurons_per_dim=40, seed=42)
    
    ssp = system['ssp']
    encoder = system['encoder']
    decoder = system['decoder']
    
    # Create networks
    print("Creating encoder network...")
    enc_net, enc_in, enc_out = encoder.create_network()
    
    print("Creating decoder network...")
    decoder.generate_training_data(n_samples=1000)
    dec_net, dec_in, dec_out = decoder.create_network()
    
    print("Computing decoders...")
    enc_in.compute_decoders(n_samples=1000)
    enc_out.compute_decoders(n_samples=1000)
    print("Done!")
    print()
    
    # Test positions
    test_positions = [
        (0.0, 0.0),
        (2.0, 1.0),
        (-1.5, -2.5),
        (3.0, -1.0),
        (-2.0, 3.5)
    ]
    
    results = []
    
    print("Testing full pipeline:")
    print("-" * 70)
    
    for x, y in test_positions:
        # Step 1: Neural encoding
        enc_in.set_input(np.array([x, y]))
        enc_net.probe(enc_out, 'encoded', 'decoded')
        enc_net.run(duration=0.3)
        
        encoded_outputs = enc_net.get_probe_data('encoded')
        encoded_ssp = encoded_outputs[-50:].mean(axis=0)
        
        # Step 2: Neural decoding
        dec_in.set_input(encoded_ssp)
        dec_net.probe(dec_out, 'decoded', 'decoded')
        dec_net.run(duration=0.3)
        
        decoded_outputs = dec_net.get_probe_data('decoded')
        decoded_pos = decoded_outputs[-50:].mean(axis=0)
        
        # Calculate error
        error = np.linalg.norm(decoded_pos - np.array([x, y]))
        
        print(f"Input:  ({x:5.1f}, {y:5.1f})")
        print(f"Output: ({decoded_pos[0]:5.2f}, {decoded_pos[1]:5.2f})")
        print(f"Error:  {error:.3f}")
        print()
        
        results.append({
            'input': (x, y),
            'output': decoded_pos,
            'error': error
        })
        
        # Reset networks
        enc_net.time = 0.0
        enc_net.probes = {}
        dec_net.time = 0.0
        dec_net.probes = {}
        for ens in enc_net.ensembles:
            ens.reset()
        for ens in dec_net.ensembles:
            ens.reset()
    
    # Summary
    avg_error = np.mean([r['error'] for r in results])
    max_error = np.max([r['error'] for r in results])
    min_error = np.min([r['error'] for r in results])
    
    print("=" * 70)
    print("Pipeline Summary:")
    print(f"  Average error: {avg_error:.3f}")
    print(f"  Max error:     {max_error:.3f}")
    print(f"  Min error:     {min_error:.3f}")
    print("=" * 70)
    print()
    
    return results


def visualize_encoding_comparison():
    """
    Create visualization comparing mathematical and neural encoding.
    """
    print("=" * 70)
    print("VISUALIZATION: Encoding Comparison")
    print("=" * 70)
    print()
    
    # Create system
    ssp = SpatialSemanticPointer(dimensions=512, seed=42)
    encoder = NeuralSSPEncoder(ssp, n_neurons_per_dim=40, seed=42)
    
    # Create network
    net, input_ens, output_ens = encoder.create_network()
    
    print("Computing decoders...")
    input_ens.compute_decoders(n_samples=1000)
    output_ens.compute_decoders(n_samples=1000)
    print()
    
    # Create grid of positions
    x_vals = np.linspace(-4, 4, 5)
    y_vals = np.linspace(-4, 4, 5)
    
    fig, axes = plt.subplots(5, 5, figsize=(15, 15))
    
    print("Generating heatmaps...")
    
    for i, x in enumerate(x_vals):
        for j, y in enumerate(y_vals):
            # Mathematical encoding
            math_ssp = ssp.encode_position(x, y)
            
            # Neural encoding
            input_ens.set_input(np.array([x, y]))
            net.probe(output_ens, 'output', 'decoded')
            net.run(duration=0.3)
            
            outputs = net.get_probe_data('output')
            neural_ssp = outputs[-50:].mean(axis=0)
            
            # Create similarity heatmap
            resolution = 20
            x_test = np.linspace(-5, 5, resolution)
            y_test = np.linspace(-5, 5, resolution)
            X, Y = np.meshgrid(x_test, y_test)
            Z = np.zeros_like(X)
            
            for ki in range(resolution):
                for kj in range(resolution):
                    test_ssp = ssp.encode_position(x_test[ki], y_test[kj])
                    Z[kj, ki] = ssp.similarity(neural_ssp, test_ssp)
            
            # Plot
            ax = axes[j, i]
            im = ax.contourf(X, Y, Z, levels=15, cmap='RdBu_r')
            ax.scatter([x], [y], c='yellow', s=50, marker='x')
            ax.set_xlim(-5, 5)
            ax.set_ylim(-5, 5)
            ax.set_aspect('equal')
            ax.set_title(f'({x:.1f}, {y:.1f})', fontsize=8)
            ax.axis('off')
            
            # Reset network
            net.time = 0.0
            net.probes = {}
            for ens in net.ensembles:
                ens.reset()
    
    plt.tight_layout()
    plt.savefig('results/figures/neural_encoding_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved: results/figures/neural_encoding_comparison.png")
    print()


def run_all_neural_experiments():
    """
    Run all neural SSP experiments.
    """
    print("\n" + "=" * 70)
    print("RUNNING ALL NEURAL SSP EXPERIMENTS")
    print("=" * 70 + "\n")
    
    # Experiment 1
    exp1_results = experiment_1_neural_encoding()
    
    # Experiment 2
    exp2_results = experiment_2_neural_decoding()
    
    # Experiment 3
    exp3_results = experiment_3_neural_transformation()
    
    # Experiment 4
    exp4_results = experiment_4_encode_decode_pipeline()
    
    # Visualization
    visualize_encoding_comparison()  # This takes a while
    
    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 70 + "\n")
    
    return {
        'encoding': exp1_results,
        'decoding': exp2_results,
        'transformation': exp3_results,
        'pipeline': exp4_results
    }


if __name__ == '__main__':
    results = run_all_neural_experiments()