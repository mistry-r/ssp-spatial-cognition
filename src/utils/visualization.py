import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def plot_heatmap(X, Y, Z, title="", ax=None, cmap='RdBu_r', vmin=-1, vmax=1):
    """
    Plot a 2D heatmap of similarities.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    im = ax.contourf(X, Y, Z, levels=20, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xlabel('x position')
    ax.set_ylabel('y position')
    ax.set_title(title)
    ax.set_aspect('equal')
    
    plt.colorbar(im, ax=ax, label='Similarity')
    
    return ax

def plot_memory_contents(memory, ax=None, marker_size=100):
    """
    Plot the actual positions of objects in memory.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot each object
    for obj_name, positions in memory.objects.items():
        positions = np.array(positions)
        if len(positions) > 0:
            ax.scatter(positions[:, 0], positions[:, 1], 
                      s=marker_size, label=obj_name, alpha=0.7)
            
            # Add labels
            for i, (x, y) in enumerate(positions):
                ax.annotate(obj_name, (x, y), 
                           xytext=(5, 5), textcoords='offset points')
    
    ax.set_xlabel('x position')
    ax.set_ylabel('y position')
    ax.set_title('Memory Contents')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    return ax

def plot_object_queries(memory, object_names, bounds=(-5, 5), resolution=50):
    """
    Create a grid of heatmaps showing where each object is located.
    """
    n_objects = len(object_names)
    n_cols = min(3, n_objects)
    n_rows = (n_objects + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    
    if n_objects == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i, obj_name in enumerate(object_names):
        X, Y, Z = memory.get_heatmap(obj_name, bounds, resolution)
        plot_heatmap(X, Y, Z, title=f"Where is {obj_name}?", ax=axes[i])
        
        # Mark true positions if available
        if obj_name in memory.objects:
            positions = np.array(memory.objects[obj_name])
            axes[i].scatter(positions[:, 0], positions[:, 1], 
                          c='yellow', s=100, marker='x', linewidths=3)
    
    # Hide unused subplots
    for i in range(n_objects, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    return fig

def plot_location_queries(memory, positions, bounds=(-5, 5)):
    """
    Query multiple locations and show bar charts of object similarities.
    """
    n_positions = len(positions)
    fig, axes = plt.subplots(1, n_positions, figsize=(5*n_positions, 4))
    
    if n_positions == 1:
        axes = [axes]
    
    for i, (x, y) in enumerate(positions):
        # Query location
        obj_name, similarity = memory.query_location(x, y)
        
        # Get similarities to all objects
        pos_ssp = memory.ssp.encode_position(x, y)
        pos_inv = memory.ssp.get_inverse(pos_ssp)
        result = memory.ssp.circular_convolution(memory.memory, pos_inv)
        
        similarities = {}
        for obj_name_vocab, obj_sp in memory.vocabulary.items():
            sim = memory.ssp.similarity(result, obj_sp)
            similarities[obj_name_vocab] = sim
        
        # Plot bar chart
        names = list(similarities.keys())
        values = list(similarities.values())
        
        axes[i].bar(names, values)
        axes[i].set_title(f"What is at ({x:.1f}, {y:.1f})?")
        axes[i].set_ylabel('Similarity')
        axes[i].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        axes[i].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    return fig