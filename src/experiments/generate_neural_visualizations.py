import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.ssp.fractional_binding import SpatialSemanticPointer
from src.ssp.spatial_memory import SpatialMemory
from src.ssp.neural_convolution import NeuralSSPEncoder, NeuralSSPDecoder, NeuralCircularConvolution
from src.utils.visualization import plot_heatmap


class NeuralSpatialMemory:
    """Neural implementation of spatial memory with query operations."""
    
    def __init__(self, ssp_generator, dimensions=512, n_neurons_per_dim=50, seed=None):
        """Initialize neural spatial memory."""
        self.ssp = ssp_generator
        self.d = dimensions
        self.n_neurons_per_dim = n_neurons_per_dim
        self.seed = seed
        
        # Mathematical memory for storing objects
        self.math_memory = SpatialMemory(ssp_generator, dimensions, seed)
        
        # Neural components
        print(f"Initializing neural encoder (this may take 1-2 minutes)...")
        self.encoder = NeuralSSPEncoder(ssp_generator, dimensions, n_neurons_per_dim, seed)
        print(f"Initializing neural decoder (this may take 1-2 minutes)...")
        self.decoder = NeuralSSPDecoder(ssp_generator, dimensions, n_neurons_per_dim, seed=seed)
        print(f"Initializing neural convolution...")
        self.convolution = NeuralCircularConvolution(dimensions, n_neurons_per_dim, seed)
        print(f"Neural components ready!\n")
        
        # Memory vector
        self.memory_vector = np.zeros(dimensions)
    
    def add_object(self, object_name, x, y):
        """Add object to memory (mathematical)."""
        self.math_memory.add_object(object_name, x, y)
        self.memory_vector = self.math_memory.memory.copy()
    
    def remove_object(self, object_name, x, y):
        """Remove object from memory."""
        self.math_memory.remove_object(object_name, x, y)
        self.memory_vector = self.math_memory.memory.copy()
    
    def neural_query_object(self, object_name, duration=0.3):
        """Neural query: Where is object_name?"""
        if object_name not in self.math_memory.vocabulary:
            return None
        
        # Normalize memory
        memory_norm = self.memory_vector / (np.linalg.norm(self.memory_vector) + 1e-10)
        
        # Get object SP and inverse
        obj_sp = self.math_memory.vocabulary[object_name]
        obj_sp = obj_sp / (np.linalg.norm(obj_sp) + 1e-10)
        obj_inv = self.ssp.get_inverse(obj_sp)
        
        # Neural convolution: M ⊛ OBJ^(-1)
        result_ssp = self.convolution.convolve(memory_norm, obj_inv, duration=duration)
        
        # Neural decoding: SSP → (x, y)
        x, y = self.decoder.decode(result_ssp, duration=duration)
        
        return x, y
    
    def neural_query_location(self, x, y, duration=0.3):
        """Neural query: What is at location (x, y)?"""
        # Normalize memory
        memory_norm = self.memory_vector / (np.linalg.norm(self.memory_vector) + 1e-10)
        
        # Neural encoding: (x, y) → SSP
        pos_ssp = self.encoder.encode(x, y, duration=duration)
        
        # Get inverse
        pos_inv = self.ssp.get_inverse(pos_ssp)
        
        # Neural convolution: M ⊛ S(x,y)^(-1)
        result = self.convolution.convolve(memory_norm, pos_inv, duration=duration)
        
        # Normalize
        result = result / (np.linalg.norm(result) + 1e-10)
        
        # Find most similar object
        similarities = {}
        for obj_name, obj_sp in self.math_memory.vocabulary.items():
            obj_sp_norm = obj_sp / (np.linalg.norm(obj_sp) + 1e-10)
            sim = self.ssp.similarity(result, obj_sp_norm)
            similarities[obj_name] = sim
        
        if len(similarities) == 0:
            return None, 0.0
        
        best_object = max(similarities.keys(), key=lambda k: similarities[k])
        best_similarity = similarities[best_object]
        
        threshold = 0.05
        if best_similarity > threshold:
            return best_object, best_similarity
        else:
            return None, best_similarity


def generate_object_query_heatmaps_comparison():
    """
    Generate side-by-side comparison of mathematical vs neural object queries.
    Similar to object_queries.png from mathematical implementation.
    """
    print("=" * 60)
    print("GENERATING OBJECT QUERY HEATMAPS COMPARISON")
    print("=" * 60)
    print()
    
    # Set up - use smaller dimensions for faster neural computation
    d = 256
    ssp = SpatialSemanticPointer(dimensions=d, seed=42)
    
    # Mathematical memory
    print("Setting up mathematical memory...")
    math_memory = SpatialMemory(ssp, dimensions=d, seed=42)
    
    # Neural memory (this will take a few minutes to initialize)
    print("Setting up neural memory...")
    neural_memory = NeuralSpatialMemory(ssp, dimensions=d, n_neurons_per_dim=30, seed=42)
    
    # Add objects to both
    objects = {
        'FOX': (1.3, -1.2),
        'DOG': (1.1, 1.7),
        'BEAR': (2.4, -2.1),
    }
    
    print("Adding objects to memories...")
    for obj_name, (x, y) in objects.items():
        math_memory.add_object(obj_name, x, y)
        neural_memory.add_object(obj_name, x, y)
    
    math_memory.normalize_memory()
    neural_memory.math_memory.normalize_memory()
    neural_memory.memory_vector = neural_memory.math_memory.memory.copy()
    
    # Create figure with 2 rows (math vs neural) x 3 columns (3 objects)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    print("\nGenerating heatmaps...")
    for idx, obj_name in enumerate(['FOX', 'DOG', 'BEAR']):
        true_x, true_y = objects[obj_name]
        
        # Mathematical heatmap (top row)
        print(f"  Mathematical query: Where is {obj_name}?")
        X, Y, Z = math_memory.get_heatmap(obj_name, bounds=(-5, 5), resolution=40)
        plot_heatmap(X, Y, Z, title=f"Mathematical: Where is {obj_name}?", ax=axes[0, idx])
        axes[0, idx].scatter([true_x], [true_y], c='yellow', s=200, marker='x', linewidths=3, label='True')
        axes[0, idx].legend()
        
        # Neural query (bottom row)
        print(f"  Neural query: Where is {obj_name}?")
        result = neural_memory.neural_query_object(obj_name, duration=0.3)
        if result:
            neural_x, neural_y = result
            error = np.sqrt((neural_x - true_x)**2 + (neural_y - true_y)**2)
            
            # For neural, show the result as a point since we can't generate full heatmap efficiently
            axes[1, idx].scatter([true_x], [true_y], c='yellow', s=200, marker='x', 
                                linewidths=3, label='True')
            axes[1, idx].scatter([neural_x], [neural_y], c='red', s=200, marker='o', 
                                alpha=0.6, label=f'Neural (err={error:.1f})')
            axes[1, idx].set_xlim(-10, 10)
            axes[1, idx].set_ylim(-10, 10)
            axes[1, idx].set_xlabel('x position')
            axes[1, idx].set_ylabel('y position')
            axes[1, idx].set_title(f"Neural: Where is {obj_name}?")
            axes[1, idx].legend()
            axes[1, idx].grid(True, alpha=0.3)
            axes[1, idx].set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('results/figures/object_queries_comparison.png', dpi=150, bbox_inches='tight')
    print("\nSaved: results/figures/object_queries_comparison.png")
    print()


def generate_location_query_comparison():
    """
    Generate comparison of mathematical vs neural location queries.
    """
    print("=" * 60)
    print("GENERATING LOCATION QUERY COMPARISON")
    print("=" * 60)
    print()
    
    d = 256
    ssp = SpatialSemanticPointer(dimensions=d, seed=42)
    
    # Set up memories
    print("Setting up memories...")
    math_memory = SpatialMemory(ssp, dimensions=d, seed=42)
    neural_memory = NeuralSpatialMemory(ssp, dimensions=d, n_neurons_per_dim=30, seed=42)
    
    # Add objects
    objects = {
        'FOX': (1.3, -1.2),
        'DOG': (1.1, 1.7),
        'BEAR': (2.4, -2.1),
        'WOLF': (-3.0, 2.5),
    }
    
    for obj_name, (x, y) in objects.items():
        math_memory.add_object(obj_name, x, y)
        neural_memory.add_object(obj_name, x, y)
    
    math_memory.normalize_memory()
    neural_memory.math_memory.normalize_memory()
    neural_memory.memory_vector = neural_memory.math_memory.memory.copy()
    
    # Test locations
    query_positions = [
        (1.3, -1.2, 'FOX'),
        (1.1, 1.7, 'DOG'),
        (-3.0, 2.5, 'WOLF'),
        (0.0, 0.0, 'Empty'),
    ]
    
    # Create comparison figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Mathematical results
    print("\nMathematical location queries:")
    math_results = []
    for x, y, expected in query_positions:
        detected, similarity = math_memory.query_location(x, y)
        math_results.append((expected, detected, similarity))
        print(f"  Location ({x:.1f}, {y:.1f}): Expected {expected}, Detected {detected}, Sim={similarity:.3f}")
    
    # Neural results
    print("\nNeural location queries:")
    neural_results = []
    for x, y, expected in query_positions:
        detected, similarity = neural_memory.neural_query_location(x, y, duration=0.3)
        neural_results.append((expected, detected, similarity))
        print(f"  Location ({x:.1f}, {y:.1f}): Expected {expected}, Detected {detected}, Sim={similarity:.3f}")
    
    # Plot bar charts
    labels = [f"({x:.1f},{y:.1f})\n{exp}" for x, y, exp in query_positions]
    
    # Mathematical
    math_sims = [r[2] for r in math_results]
    axes[0].bar(range(len(labels)), math_sims, color='blue', alpha=0.7)
    axes[0].set_xticks(range(len(labels)))
    axes[0].set_xticklabels(labels, rotation=0, ha='center')
    axes[0].set_ylabel('Similarity')
    axes[0].set_title('Mathematical Location Query')
    axes[0].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    axes[0].grid(True, alpha=0.3)
    
    # Neural
    neural_sims = [r[2] for r in neural_results]
    axes[1].bar(range(len(labels)), neural_sims, color='red', alpha=0.7)
    axes[1].set_xticks(range(len(labels)))
    axes[1].set_xticklabels(labels, rotation=0, ha='center')
    axes[1].set_ylabel('Similarity')
    axes[1].set_title('Neural Location Query')
    axes[1].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/figures/location_queries_comparison.png', dpi=150, bbox_inches='tight')
    print("\nSaved: results/figures/location_queries_comparison.png")
    print()


def generate_memory_retention_comparison():
    """
    Compare memory retention behavior between mathematical and neural implementations.
    """
    print("=" * 60)
    print("GENERATING MEMORY RETENTION COMPARISON")
    print("=" * 60)
    print()
    
    d = 256
    ssp = SpatialSemanticPointer(dimensions=d, seed=42)
    
    # Create two pairs of memories (full and partial)
    print("Setting up memories...")
    math_full = SpatialMemory(ssp, dimensions=d, seed=42)
    math_partial = SpatialMemory(ssp, dimensions=d, seed=42)
    neural_full = NeuralSpatialMemory(ssp, dimensions=d, n_neurons_per_dim=30, seed=42)
    neural_partial = NeuralSpatialMemory(ssp, dimensions=d, n_neurons_per_dim=30, seed=42)
    
    # Initial configuration
    initial_objects = {
        'A': (0, 0),
        'B': (2, 2),
        'C': (-2, -2),
        'D': (2, -2),
        'E': (-2, 2)
    }
    
    print("Adding objects...")
    for obj, (x, y) in initial_objects.items():
        math_full.add_object(obj, x, y)
        math_partial.add_object(obj, x, y)
        neural_full.add_object(obj, x, y)
        neural_partial.add_object(obj, x, y)
    
    # Remove objects from partial memories
    removed_objects = ['B', 'D']
    print(f"Removing objects {removed_objects} from partial memories...")
    for obj in removed_objects:
        x, y = initial_objects[obj]
        math_partial.remove_object(obj, x, y)
        neural_partial.remove_object(obj, x, y)
    
    math_full.normalize_memory()
    math_partial.normalize_memory()
    neural_full.math_memory.normalize_memory()
    neural_full.memory_vector = neural_full.math_memory.memory.copy()
    neural_partial.math_memory.normalize_memory()
    neural_partial.memory_vector = neural_partial.math_memory.memory.copy()
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    print("\nTesting memory retention...")
    
    # Row 1: Mathematical
    print("Mathematical - querying removed object B at (2, 2)...")
    math_full_pos = math_full.query_object('B', bounds=(-4, 4), resolution=30)
    if math_full_pos:
        axes[0, 0].scatter([math_full_pos[0][0]], [math_full_pos[0][1]], 
                          c='green', s=200, marker='o', label='Full memory query')
    axes[0, 0].scatter([2], [2], c='yellow', s=200, marker='x', linewidths=3, label='True location')
    axes[0, 0].set_xlim(-10, 10)
    axes[0, 0].set_ylim(-10, 10)
    axes[0, 0].set_title('Mathematical: Full Memory (B present)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_aspect('equal')
    
    math_partial_pos = math_partial.query_object('B', bounds=(-4, 4), resolution=30)
    if math_partial_pos:
        axes[0, 1].scatter([math_partial_pos[0][0]], [math_partial_pos[0][1]], 
                          c='orange', s=200, marker='o', label='Partial memory query')
    axes[0, 1].scatter([2], [2], c='red', s=200, marker='x', linewidths=3, label='Removed location')
    axes[0, 1].set_xlim(-10, 10)
    axes[0, 1].set_ylim(-10, 10)
    axes[0, 1].set_title('Mathematical: Partial Memory (B removed)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_aspect('equal')
    
    # Row 2: Neural
    print("Neural - querying removed object B at (2, 2)...")
    neural_full_result = neural_full.neural_query_object('B', duration=0.3)
    if neural_full_result:
        neural_x, neural_y = neural_full_result
        axes[1, 0].scatter([neural_x], [neural_y], c='green', s=200, marker='o', 
                          label='Full memory query')
    axes[1, 0].scatter([2], [2], c='yellow', s=200, marker='x', linewidths=3, label='True location')
    axes[1, 0].set_xlim(-10, 10)
    axes[1, 0].set_ylim(-10, 10)
    axes[1, 0].set_title('Neural: Full Memory (B present)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_aspect('equal')
    
    neural_partial_result = neural_partial.neural_query_object('B', duration=0.3)
    if neural_partial_result:
        neural_x, neural_y = neural_partial_result
        axes[1, 1].scatter([neural_x], [neural_y], c='orange', s=200, marker='o', 
                          label='Partial memory query')
    axes[1, 1].scatter([2], [2], c='red', s=200, marker='x', linewidths=3, label='Removed location')
    axes[1, 1].set_xlim(-10, 10)
    axes[1, 1].set_ylim(-10, 10)
    axes[1, 1].set_title('Neural: Partial Memory (B removed)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('results/figures/memory_retention_comparison.png', dpi=150, bbox_inches='tight')
    print("\nSaved: results/figures/memory_retention_comparison.png")
    print()


def generate_all_comparisons():
    """Generate all comparison visualizations."""
    print("\n" + "=" * 60)
    print("GENERATING MATHEMATICAL VS NEURAL VISUALIZATIONS")
    print("=" * 60)
    print("\nNOTE: Neural initialization takes 2-3 minutes per memory system.")
    print("Total time: ~10-15 minutes for all visualizations.\n")
    
    # Create results directory if it doesn't exist
    os.makedirs('results/figures', exist_ok=True)
    
    # Generate comparisons
    generate_object_query_heatmaps_comparison()
    generate_location_query_comparison()
    generate_memory_retention_comparison()
    
    # Generate standalone neural plots
    generate_neural_only_visualizations()
    
    print("\n" + "=" * 60)
    print("ALL VISUALIZATIONS COMPLETE")
    print("=" * 60)
    print("\nGenerated comparison files:")
    print("  - results/figures/object_queries_comparison.png")
    print("  - results/figures/location_queries_comparison.png")
    print("  - results/figures/memory_retention_comparison.png")
    print("\nGenerated neural-only files:")
    print("  - results/figures/neural_object_queries.png")
    print("  - results/figures/neural_location_queries.png")
    print("  - results/figures/neural_memory_retention.png")
    print("  - results/figures/neural_memory_contents.png")
    print("\nThese visualizations show side-by-side comparison and")
    print("standalone neural SSP implementation results.")
    print("=" * 60 + "\n")


def generate_neural_only_visualizations():
    """Generate standalone neural visualizations (not comparisons)."""
    print("\n" + "=" * 60)
    print("GENERATING STANDALONE NEURAL VISUALIZATIONS")
    print("=" * 60)
    print()
    
    d = 256
    ssp = SpatialSemanticPointer(dimensions=d, seed=42)
    
    print("Setting up neural memory...")
    neural_memory = NeuralSpatialMemory(ssp, dimensions=d, n_neurons_per_dim=30, seed=42)
    
    # Add objects
    objects = {
        'FOX': (1.3, -1.2),
        'DOG': (1.1, 1.7),
        'BEAR': (2.4, -2.1),
        'WOLF': (-3.0, 2.5),
        'BADGER': (-1.5, -3.0)
    }
    
    print("Adding objects to neural memory...")
    for obj_name, (x, y) in objects.items():
        neural_memory.add_object(obj_name, x, y)
    
    neural_memory.math_memory.normalize_memory()
    neural_memory.memory_vector = neural_memory.math_memory.memory.copy()
    
    # 1. Neural Memory Contents
    print("\nGenerating neural memory contents plot...")
    fig1, ax1 = plt.subplots(figsize=(8, 8))
    
    # Plot actual positions
    for obj_name, (x, y) in objects.items():
        ax1.scatter([x], [y], s=100, alpha=0.7, label=obj_name)
        ax1.annotate(obj_name, (x, y), xytext=(5, 5), textcoords='offset points')
    
    ax1.set_xlabel('x position')
    ax1.set_ylabel('y position')
    ax1.set_title('Neural Memory Contents\n(Stored Objects)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    ax1.set_xlim(-10, 10)
    ax1.set_ylim(-10, 10)
    
    plt.tight_layout()
    plt.savefig('results/figures/neural_memory_contents.png', dpi=150, bbox_inches='tight')
    print("Saved: results/figures/neural_memory_contents.png")
    
    # 2. Neural Object Queries
    print("\nGenerating neural object queries plot...")
    fig2, axes2 = plt.subplots(2, 3, figsize=(15, 10))
    axes2 = axes2.flatten()
    
    query_objects = ['FOX', 'DOG', 'BEAR', 'WOLF', 'BADGER']
    
    for idx, obj_name in enumerate(query_objects):
        true_x, true_y = objects[obj_name]
        
        print(f"  Querying: Where is {obj_name}?")
        result = neural_memory.neural_query_object(obj_name, duration=0.3)
        
        if result:
            neural_x, neural_y = result
            error = np.sqrt((neural_x - true_x)**2 + (neural_y - true_y)**2)
            
            # Plot
            axes2[idx].scatter([true_x], [true_y], c='green', s=200, marker='x', 
                             linewidths=3, label='True location')
            axes2[idx].scatter([neural_x], [neural_y], c='red', s=200, marker='o', 
                             alpha=0.6, label=f'Neural query')
            
            # Draw arrow from neural to true
            axes2[idx].arrow(neural_x, neural_y, true_x - neural_x, true_y - neural_y,
                           head_width=0.2, head_length=0.15, fc='gray', ec='gray', 
                           alpha=0.5, linestyle='--')
            
            axes2[idx].set_xlim(-10, 10)
            axes2[idx].set_ylim(-10, 10)
            axes2[idx].set_xlabel('x position')
            axes2[idx].set_ylabel('y position')
            axes2[idx].set_title(f"Where is {obj_name}?\nError: {error:.2f} units")
            axes2[idx].legend()
            axes2[idx].grid(True, alpha=0.3)
            axes2[idx].set_aspect('equal')
    
    # Hide the 6th subplot
    axes2[5].axis('off')
    
    plt.tight_layout()
    plt.savefig('results/figures/neural_object_queries.png', dpi=150, bbox_inches='tight')
    print("Saved: results/figures/neural_object_queries.png")
    
    # 3. Neural Location Queries
    print("\nGenerating neural location queries plot...")
    query_positions = [
        (1.3, -1.2, 'FOX'),
        (1.1, 1.7, 'DOG'),
        (-3.0, 2.5, 'WOLF'),
        (0.0, 0.0, 'Empty'),
    ]
    
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    
    labels = []
    similarities = []
    colors = []
    
    print("\nNeural location queries:")
    for x, y, expected in query_positions:
        detected, similarity = neural_memory.neural_query_location(x, y, duration=0.3)
        
        label = f"({x:.1f},{y:.1f})\n{expected}"
        labels.append(label)
        similarities.append(similarity)
        
        # Color based on correctness
        if detected == expected:
            colors.append('green')
        else:
            colors.append('red')
        
        print(f"  Location ({x:.1f}, {y:.1f}): Expected {expected}, Detected {detected}, Sim={similarity:.3f}")
    
    ax3.bar(range(len(labels)), similarities, color=colors, alpha=0.7)
    ax3.set_xticks(range(len(labels)))
    ax3.set_xticklabels(labels, rotation=0, ha='center')
    ax3.set_ylabel('Similarity')
    ax3.set_title('Neural Location Query Results\n(Green=Correct, Red=Incorrect)')
    ax3.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax3.axhline(y=0.05, color='orange', linestyle='--', linewidth=1, label='Threshold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/figures/neural_location_queries.png', dpi=150, bbox_inches='tight')
    print("Saved: results/figures/neural_location_queries.png")
    
    # 4. Neural Memory Retention
    print("\nGenerating neural memory retention plot...")
    
    # Create two neural memories
    neural_full = NeuralSpatialMemory(ssp, dimensions=d, n_neurons_per_dim=30, seed=43)
    neural_partial = NeuralSpatialMemory(ssp, dimensions=d, n_neurons_per_dim=30, seed=44)
    
    retention_objects = {
        'A': (0, 0),
        'B': (2, 2),
        'C': (-2, -2),
        'D': (2, -2),
        'E': (-2, 2)
    }
    
    print("Testing neural memory retention...")
    for obj, (x, y) in retention_objects.items():
        neural_full.add_object(obj, x, y)
        neural_partial.add_object(obj, x, y)
    
    # Remove B and D from partial
    removed = ['B', 'D']
    for obj in removed:
        x, y = retention_objects[obj]
        neural_partial.remove_object(obj, x, y)
    
    neural_full.math_memory.normalize_memory()
    neural_full.memory_vector = neural_full.math_memory.memory.copy()
    neural_partial.math_memory.normalize_memory()
    neural_partial.memory_vector = neural_partial.math_memory.memory.copy()
    
    fig4, axes4 = plt.subplots(1, 2, figsize=(14, 6))
    
    # Full memory
    print("  Querying B in full memory...")
    result_full = neural_full.neural_query_object('B', duration=0.3)
    if result_full:
        x_full, y_full = result_full
        error_full = np.sqrt((x_full - 2)**2 + (y_full - 2)**2)
        axes4[0].scatter([2], [2], c='green', s=200, marker='x', linewidths=3, label='True location')
        axes4[0].scatter([x_full], [y_full], c='blue', s=200, marker='o', alpha=0.6, 
                        label=f'Query result (err={error_full:.1f})')
    
    axes4[0].set_xlim(-10, 10)
    axes4[0].set_ylim(-10, 10)
    axes4[0].set_xlabel('x position')
    axes4[0].set_ylabel('y position')
    axes4[0].set_title('Neural: Full Memory\n(B present)')
    axes4[0].legend()
    axes4[0].grid(True, alpha=0.3)
    axes4[0].set_aspect('equal')
    
    # Partial memory (B removed)
    print("  Querying B in partial memory (after removal)...")
    result_partial = neural_partial.neural_query_object('B', duration=0.3)
    if result_partial:
        x_partial, y_partial = result_partial
        error_partial = np.sqrt((x_partial - 2)**2 + (y_partial - 2)**2)
        axes4[1].scatter([2], [2], c='red', s=200, marker='x', linewidths=3, 
                        label='Removed location')
        axes4[1].scatter([x_partial], [y_partial], c='orange', s=200, marker='o', alpha=0.6, 
                        label=f'Query result (err={error_partial:.1f})')
    
    axes4[1].set_xlim(-10, 10)
    axes4[1].set_ylim(-10, 10)
    axes4[1].set_xlabel('x position')
    axes4[1].set_ylabel('y position')
    axes4[1].set_title('Neural: Partial Memory\n(B removed - shows residual trace)')
    axes4[1].legend()
    axes4[1].grid(True, alpha=0.3)
    axes4[1].set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('results/figures/neural_memory_retention.png', dpi=150, bbox_inches='tight')
    print("Saved: results/figures/neural_memory_retention.png")
    print()


if __name__ == '__main__':
    generate_all_comparisons()