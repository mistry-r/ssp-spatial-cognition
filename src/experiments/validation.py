import numpy as np
import matplotlib.pyplot as plt
from ..ssp.fractional_binding import SpatialSemanticPointer
from ..ssp.spatial_memory import SpatialMemory
from .object_detection import (
    test_single_object_query,
    test_missing_object_query,
    test_location_query,
    test_duplicate_object_query,
    test_construct_and_readout_ssp
)
from .neural_queries import (
    test_neural_encoding_decoding,
    test_neural_object_query,
    test_neural_location_query,
    test_neural_convolution
)

def generate_table_2_comparison(n_trials=100, dimensions=512):
    """
    Generate complete comparison to Table 2 from the paper.
    
    Runs both mathematical and neural versions of all tests.
    """
    print("\n" + "=" * 70)
    print("REPLICATION OF TABLE 2: Komer et al. (2019)")
    print("=" * 70 + "\n")
    
    results = {
        'desiderata': [],
        'non_neural_accuracy': [],
        'non_neural_target': [],
        'neural_accuracy': [],
        'neural_target': []
    }
    
    # Test 1: Query single object
    print("Running: Query single object...")
    non_neural = test_single_object_query(n_trials=n_trials, dimensions=dimensions)
    neural = test_neural_object_query(n_trials=max(50, n_trials//2), dimensions=dimensions)
    
    results['desiderata'].append('Query single object')
    results['non_neural_accuracy'].append(non_neural)
    results['non_neural_target'].append(99.1)
    results['neural_accuracy'].append(neural)
    results['neural_target'].append(95.7)
    
    # Test 2: Query missing object
    print("Running: Query missing object...")
    non_neural = test_missing_object_query(n_trials=n_trials, dimensions=dimensions)
    
    results['desiderata'].append('Query missing object')
    results['non_neural_accuracy'].append(non_neural)
    results['non_neural_target'].append(99.4)
    results['neural_accuracy'].append('N/A')  # Not in neural tests yet
    results['neural_target'].append(96.7)
    
    # Test 3: Query location
    print("Running: Query location...")
    non_neural = test_location_query(n_trials=n_trials, dimensions=dimensions)
    neural = test_neural_location_query(n_trials=max(50, n_trials//2), dimensions=dimensions)
    
    results['desiderata'].append('Query location')
    results['non_neural_accuracy'].append(non_neural)
    results['non_neural_target'].append(97.3)
    results['neural_accuracy'].append(neural)
    results['neural_target'].append(94.7)
    
    # Test 4: Query duplicate object
    print("Running: Query duplicate object...")
    non_neural = test_duplicate_object_query(n_trials=n_trials, dimensions=dimensions)
    
    results['desiderata'].append('Query duplicate object')
    results['non_neural_accuracy'].append(non_neural)
    results['non_neural_target'].append(97.4)
    results['neural_accuracy'].append('N/A')
    results['neural_target'].append(95.3)
    
    # Test 5: Construct and readout SSP
    print("Running: Construct and readout SSP...")
    construct_readout = test_construct_and_readout_ssp(n_trials=n_trials, dimensions=dimensions)
    encoding_decoding = test_neural_encoding_decoding(n_trials=max(50, n_trials//2), dimensions=dimensions)
    
    results['desiderata'].append('Construct SSP')
    results['non_neural_accuracy'].append(construct_readout['construct'])
    results['non_neural_target'].append(100.0)
    results['neural_accuracy'].append('N/A')  # Always perfect in construction
    results['neural_target'].append(99.0)
    
    results['desiderata'].append('Readout location from SSP')
    results['non_neural_accuracy'].append(construct_readout['readout'])
    results['non_neural_target'].append(100.0)
    results['neural_accuracy'].append(encoding_decoding)
    results['neural_target'].append(94.1)
    
    # Print table
    print("\n" + "=" * 70)
    print("TABLE 2 COMPARISON")
    print("=" * 70)
    print(f"{'Desiderata':<30} {'Non-Neural':<15} {'Target':<10} {'Neural':<15} {'Target':<10}")
    print("-" * 70)
    
    for i in range(len(results['desiderata'])):
        nn_acc = results['non_neural_accuracy'][i]
        nn_tgt = results['non_neural_target'][i]
        n_acc = results['neural_accuracy'][i]
        n_tgt = results['neural_target'][i]
        
        nn_str = f"{nn_acc:.1f}%" if isinstance(nn_acc, (int, float)) else nn_acc
        n_str = f"{n_acc:.1f}%" if isinstance(n_acc, (int, float)) else n_acc
        
        print(f"{results['desiderata'][i]:<30} {nn_str:<15} {nn_tgt:<10.1f}% {n_str:<15} {n_tgt:<10.1f}%")
    
    print("=" * 70)
    print()
    
    return results


def plot_capacity_curves(dimensions_list=[128, 256, 512], max_items=100):
    """
    Generate capacity curves (Figure 4 from paper).
    """
    print("=" * 60)
    print("GENERATING CAPACITY CURVES (Figure 4)")
    print("=" * 60)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    for dim in dimensions_list:
        print(f"Computing capacity for {dim} dimensions...")
        
        n_items_list = np.logspace(0, np.log10(max_items), 20).astype(int)
        n_items_list = np.unique(n_items_list)
        
        location_similarities = []
        object_similarities = []
        location_accuracies = []
        object_accuracies = []
        
        for n_items in n_items_list:
            ssp = SpatialSemanticPointer(dimensions=dim, seed=42)
            memory = SpatialMemory(ssp, dimensions=dim, seed=42)
            
            # Add objects
            positions = {}
            for i in range(n_items):
                obj_name = f"OBJ_{i}"
                x = np.random.uniform(-5, 5)
                y = np.random.uniform(-5, 5)
                memory.add_object(obj_name, x, y)
                positions[obj_name] = (x, y)
            
            memory.normalize_memory()
            
            # Test location query
            query_obj = f"OBJ_0"
            query_x, query_y = positions[query_obj]
            result_obj, similarity = memory.query_location(query_x, query_y)
            
            location_similarities.append(similarity)
            location_accuracies.append(1.0 if result_obj == query_obj else 0.0)
            
            # Test object query
            decoded_positions = memory.query_object(query_obj, bounds=(-5, 5), resolution=100)
            if len(decoded_positions) > 0:
                decoded_pos = decoded_positions[0]
                error = np.linalg.norm(np.array(decoded_pos) - np.array(positions[query_obj]))
                object_accuracies.append(1.0 if error < 0.5 else 0.0)
                
                # Compute similarity for object query
                obj_sp = memory.vocabulary[query_obj]
                obj_inv = ssp.get_inverse(obj_sp)
                result_ssp = ssp.circular_convolution(memory.memory, obj_inv)
                true_pos_ssp = ssp.encode_position(query_x, query_y)
                sim = ssp.similarity(result_ssp, true_pos_ssp)
                object_similarities.append(sim)
            else:
                object_accuracies.append(0.0)
                object_similarities.append(0.0)
        
        # Plot
        label = f"{dim}D"
        
        # Location query capacity (similarity)
        axes[0, 0].semilogx(n_items_list, location_similarities, 
                            marker='o', label=label, alpha=0.7)
        
        # Object query capacity (similarity)
        axes[0, 1].semilogx(n_items_list, object_similarities,
                           marker='o', label=label, alpha=0.7)
        
        # Location query accuracy
        axes[1, 0].semilogx(n_items_list, location_accuracies,
                           marker='o', label=label, alpha=0.7)
        
        # Object query accuracy
        axes[1, 1].semilogx(n_items_list, object_accuracies,
                           marker='o', label=label, alpha=0.7)
    
    # Add threshold line
    threshold = 0.154  # 3-sigma threshold for 512D
    axes[0, 0].axhline(threshold, color='k', linestyle='--', alpha=0.3, label='Threshold')
    axes[0, 1].axhline(threshold, color='k', linestyle='--', alpha=0.3)
    
    # Formatting
    axes[0, 0].set_title('Location Query Capacity')
    axes[0, 0].set_ylabel('Similarity')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_title('Item Query Capacity')
    axes[0, 1].set_ylabel('Similarity')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_title('Location Query Accuracy')
    axes[1, 0].set_xlabel('Number of Stored Items')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_title('Item Query Accuracy')
    axes[1, 1].set_xlabel('Number of Stored Items')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/figures/capacity_curves.png', dpi=150, bbox_inches='tight')
    print("Saved: results/figures/capacity_curves.png")
    print()
    
    return fig


def run_full_validation():
    """
    Run complete validation suite.
    """
    print("\n" + "=" * 70)
    print("FULL VALIDATION SUITE")
    print("Replicating Komer et al. (2019) - Table 2 and Figure 4")
    print("=" * 70 + "\n")
    
    # Table 2 comparison
    table_results = generate_table_2_comparison(n_trials=100, dimensions=512)
    
    # Figure 4 capacity curves
    capacity_fig = plot_capacity_curves(dimensions_list=[128, 256, 512], max_items=100)
    
    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)
    print("Check results/figures/ for generated plots")
    print("=" * 70 + "\n")
    
    return table_results, capacity_fig


if __name__ == '__main__':
    results, fig = run_full_validation()