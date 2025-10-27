import numpy as np
import matplotlib.pyplot as plt
from src.ssp.fractional_binding import SpatialSemanticPointer
from src.ssp.spatial_memory import SpatialMemory
from src.utils.visualization import plot_memory_contents, plot_heatmap

def test_memory_trace():
    """
    Demonstrate that removed objects leave traces in memory.
    """
    print("=" * 60)
    print("MEMORY RETENTION EXPERIMENT")
    print("=" * 60)
    print()
    
    # Set up
    ssp = SpatialSemanticPointer(dimensions=512, seed=42)
    memory_full = SpatialMemory(ssp, dimensions=512, seed=42)
    memory_partial = SpatialMemory(ssp, dimensions=512, seed=42)
    
    # Initial configuration - 5 objects
    initial_objects = {
        'A': (0, 0),
        'B': (2, 2),
        'C': (-2, -2),
        'D': (2, -2),
        'E': (-2, 2)
    }
    
    print("Step 1: Adding 5 objects to both memories")
    for obj, (x, y) in initial_objects.items():
        memory_full.add_object(obj, x, y)
        memory_partial.add_object(obj, x, y)
    
    # Remove some objects from partial memory
    removed_objects = ['B', 'D']
    print(f"Step 2: Removing objects {removed_objects} from partial memory")
    for obj in removed_objects:
        x, y = initial_objects[obj]
        memory_partial.remove_object(obj, x, y)
    
    print()
    
    # Test 1: Query removed object locations in partial memory
    print("Test 1: Can we still detect removed objects?")
    for obj in removed_objects:
        x, y = initial_objects[obj]
        detected, similarity = memory_partial.query_location(x, y)
        print(f"  Query location of removed '{obj}' at ({x}, {y}):")
        print(f"    Detected: {detected}, Similarity: {similarity:.3f}")
    
    print()
    
    # Test 2: Compare memory similarity at removed locations
    print("Test 2: Memory interference at removed locations")
    for obj in removed_objects:
        x, y = initial_objects[obj]
        
        # Create position SSP
        pos_ssp = ssp.encode_position(x, y)
        
        # Compare similarity of both memories to this position
        sim_full = ssp.similarity(memory_full.memory, pos_ssp)
        sim_partial = ssp.similarity(memory_partial.memory, pos_ssp)
        
        print(f"Location of removed '{obj}' at ({x}, {y}):")
        print(f"Full memory similarity: {sim_full:.3f}")
        print(f"Partial memory similarity: {sim_partial:.3f}")
        print(f"Residual: {abs(sim_partial):.3f}")
    
    print()
    
    # Test 3: Heatmap comparison
    print("Test 3: Generating Visualization")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Row 1: Full memory
    axes[0, 0].set_title("Full Memory Contents")
    plot_memory_contents(memory_full, ax=axes[0, 0])
    axes[0, 0].set_xlim(-4, 4)
    axes[0, 0].set_ylim(-4, 4)
    
    X, Y, Z = memory_full.get_heatmap('B', bounds=(-4, 4), resolution=50)
    plot_heatmap(X, Y, Z, title="Full Memory: Where is B?", ax=axes[0, 1])
    axes[0, 1].scatter([2], [2], c='yellow', s=200, marker='x', linewidths=3)
    
    X, Y, Z = memory_full.get_heatmap('D', bounds=(-4, 4), resolution=50)
    plot_heatmap(X, Y, Z, title="Full Memory: Where is D?", ax=axes[0, 2])
    axes[0, 2].scatter([2], [-2], c='yellow', s=200, marker='x', linewidths=3)
    
    # Row 2: Partial memory (after removal)
    axes[1, 0].set_title("Partial Memory Contents (B, D removed)")
    plot_memory_contents(memory_partial, ax=axes[1, 0])
    axes[1, 0].scatter([2, 2], [2, -2], c='red', s=200, marker='x', 
                      linewidths=3, label='Removed')
    axes[1, 0].legend()
    axes[1, 0].set_xlim(-4, 4)
    axes[1, 0].set_ylim(-4, 4)
    
    X, Y, Z = memory_partial.get_heatmap('B', bounds=(-4, 4), resolution=50)
    plot_heatmap(X, Y, Z, title="Partial Memory: Where is B? (removed)", ax=axes[1, 1])
    axes[1, 1].scatter([2], [2], c='red', s=200, marker='x', linewidths=3)
    
    X, Y, Z = memory_partial.get_heatmap('D', bounds=(-4, 4), resolution=50)
    plot_heatmap(X, Y, Z, title="Partial Memory: Where is D? (removed)", ax=axes[1, 2])
    axes[1, 2].scatter([2], [-2], c='red', s=200, marker='x', linewidths=3)
    
    plt.tight_layout()
    plt.savefig('results/figures/memory_retention.png', dpi=150, bbox_inches='tight')
    print("Saved: results/figures/memory_retention.png")
    print()
    
    # Test 4: Quantify memory traces
    print("Test 4: Quantifying memory persistence")
    print()
    print("This demonstrates 'graceful degradation' - removed objects")
    print("leave traces that interfere with clean readout.") 
    print()

if __name__ == '__main__':    
    test_memory_trace()