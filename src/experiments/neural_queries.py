import numpy as np
from ..ssp.fractional_binding import SpatialSemanticPointer
from ..ssp.spatial_memory import SpatialMemory
from ..ssp.neural_convolution import (
    NeuralCircularConvolution,
    NeuralSSPEncoder,
    NeuralSSPDecoder
)


class NeuralSpatialMemory:
    """
    Neural implementation of spatial memory with query operations.
    """
    
    def __init__(self, ssp_generator, dimensions=512, n_neurons_per_dim=50, seed=None):
        """Initialize neural spatial memory."""
        self.ssp = ssp_generator
        self.d = dimensions
        self.n_neurons_per_dim = n_neurons_per_dim
        self.seed = seed
        
        # Mathematical memory for storing objects
        self.math_memory = SpatialMemory(ssp_generator, dimensions, seed)
        
        # Neural components
        self.encoder = NeuralSSPEncoder(ssp_generator, dimensions, n_neurons_per_dim, seed)
        self.decoder = NeuralSSPDecoder(ssp_generator, dimensions, n_neurons_per_dim, seed=seed)
        self.convolution = NeuralCircularConvolution(dimensions, n_neurons_per_dim, seed)
        
        # Memory vector
        self.memory_vector = np.zeros(dimensions)
    
    def add_object(self, object_name, x, y):
        """Add object to memory (mathematical)."""
        self.math_memory.add_object(object_name, x, y)
        self.memory_vector = self.math_memory.memory.copy()
    
    def neural_query_object(self, object_name, duration=0.3):
        """
        Neural query: Where is object_name?
        
        Implements: M ⊛ OBJ^(-1) → position SSP → (x, y)
        """
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
        """
        Neural query: What is at location (x, y)?
        
        Implements: M ⊛ S(x,y)^(-1) → object SP
        """
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


# Standalone test functions for validation

def test_neural_convolution(n_trials=20, dimensions=256):
    """Test neural circular convolution."""
    print("=" * 60)
    print("TEST: Neural Circular Convolution")
    print("=" * 60)
    
    ssp = SpatialSemanticPointer(dimensions=dimensions, seed=42)
    conv = NeuralCircularConvolution(dimensions=dimensions, n_neurons_per_dim=30, seed=42)
    
    errors = []
    
    for trial in range(n_trials):
        # Random vectors
        a = np.random.randn(dimensions)
        a = a / np.linalg.norm(a)
        b = np.random.randn(dimensions)
        b = b / np.linalg.norm(b)
        
        # Mathematical
        math_result = ssp.circular_convolution(a, b)
        math_result = math_result / np.linalg.norm(math_result)
        
        # Neural
        neural_result = conv.convolve(a, b, duration=0.3)
        
        # Similarity
        similarity = np.dot(math_result, neural_result)
        errors.append(1.0 - similarity)
        
        if trial < 3:
            print(f"Trial {trial}: Similarity = {similarity:.3f}")
    
    mean_sim = 1.0 - np.mean(errors)
    print(f"\nMean Similarity: {mean_sim:.3f}")
    print(f"Target: >0.85")
    print()
    
    return mean_sim


def test_neural_encoding_decoding(n_trials=50, dimensions=256):
    """Test neural SSP encoding/decoding."""
    print("=" * 60)
    print("TEST: Neural SSP Encoding/Decoding")
    print("=" * 60)
    
    ssp = SpatialSemanticPointer(dimensions=dimensions, seed=42)
    encoder = NeuralSSPEncoder(ssp, dimensions=dimensions, n_neurons_per_dim=30, seed=42)
    decoder = NeuralSSPDecoder(ssp, dimensions=dimensions, n_neurons_per_dim=30, seed=42)
    
    errors = []
    
    for trial in range(n_trials):
        # Random position
        true_x = np.random.uniform(-5, 5)
        true_y = np.random.uniform(-5, 5)
        
        # Encode
        ssp_neural = encoder.encode(true_x, true_y, duration=0.3)
        
        # Decode
        decoded_x, decoded_y = decoder.decode(ssp_neural, duration=0.3)
        
        # Error
        error = np.sqrt((decoded_x - true_x)**2 + (decoded_y - true_y)**2)
        errors.append(error)
        
        if trial < 3:
            print(f"Trial {trial}: True=({true_x:.2f}, {true_y:.2f}), "
                  f"Decoded=({decoded_x:.2f}, {decoded_y:.2f}), Error={error:.3f}")
    
    accuracy = np.mean(np.array(errors) < 0.5) * 100
    print(f"\nMean Error: {np.mean(errors):.3f}")
    print(f"Accuracy (<0.5 units): {accuracy:.1f}%")
    print(f"Target: ~90-94%")
    print()
    
    return accuracy


def test_neural_object_query(n_trials=20, dimensions=256):
    """Test neural object queries."""
    print("=" * 60)
    print("TEST: Neural Object Query")
    print("=" * 60)
    
    accuracies = []
    
    for trial in range(n_trials):
        n_objects = np.random.randint(2, 6)
        
        ssp = SpatialSemanticPointer(dimensions=dimensions, seed=trial)
        memory = NeuralSpatialMemory(ssp, dimensions=dimensions, n_neurons_per_dim=30, seed=trial)
        
        # Add objects
        positions = {}
        for i in range(n_objects):
            obj_name = f"OBJ_{i}"
            x = np.random.uniform(-3, 3)
            y = np.random.uniform(-3, 3)
            memory.add_object(obj_name, x, y)
            positions[obj_name] = (x, y)
        
        memory.math_memory.normalize_memory()
        memory.memory_vector = memory.math_memory.memory.copy()
        
        # Query random object
        query_obj = np.random.choice(list(positions.keys()))
        true_x, true_y = positions[query_obj]
        
        # Neural query
        result = memory.neural_query_object(query_obj, duration=0.3)
        
        if result is not None:
            decoded_x, decoded_y = result
            error = np.sqrt((decoded_x - true_x)**2 + (decoded_y - true_y)**2)
            accuracies.append(error < 1.0)
            
            if trial < 3:
                print(f"Trial {trial}: {query_obj} at ({true_x:.2f}, {true_y:.2f}), "
                      f"decoded ({decoded_x:.2f}, {decoded_y:.2f}), error={error:.3f}")
        else:
            accuracies.append(False)
    
    accuracy = np.mean(accuracies) * 100
    print(f"\nAccuracy: {accuracy:.1f}%")
    print(f"Target: ~85-95%")
    print()
    
    return accuracy


def test_neural_location_query(n_trials=20, dimensions=256):
    """Test neural location queries."""
    print("=" * 60)
    print("TEST: Neural Location Query")
    print("=" * 60)
    
    accuracies = []
    
    for trial in range(n_trials):
        n_objects = np.random.randint(2, 6)
        
        ssp = SpatialSemanticPointer(dimensions=dimensions, seed=trial)
        memory = NeuralSpatialMemory(ssp, dimensions=dimensions, n_neurons_per_dim=30, seed=trial)
        
        # Add objects
        positions = {}
        for i in range(n_objects):
            obj_name = f"OBJ_{i}"
            x = np.random.uniform(-3, 3)
            y = np.random.uniform(-3, 3)
            memory.add_object(obj_name, x, y)
            positions[obj_name] = (x, y)
        
        memory.math_memory.normalize_memory()
        memory.memory_vector = memory.math_memory.memory.copy()
        
        # Query random object's location
        query_obj = np.random.choice(list(positions.keys()))
        query_x, query_y = positions[query_obj]
        
        # Neural query
        result_obj, similarity = memory.neural_query_location(query_x, query_y, duration=0.3)
        
        accuracies.append(result_obj == query_obj)
        
        if trial < 3:
            print(f"Trial {trial}: Location ({query_x:.2f}, {query_y:.2f}) has {query_obj}, "
                  f"detected {result_obj}, similarity={similarity:.3f}")
    
    accuracy = np.mean(accuracies) * 100
    print(f"\nAccuracy: {accuracy:.1f}%")
    print(f"Target: ~85-94%")
    print()
    
    return accuracy


def run_all_neural_tests():
    """Run all neural SSP tests."""
    print("\n" + "=" * 60)
    print("RUNNING ALL NEURAL SSP TESTS")
    print("=" * 60 + "\n")
    
    results = {}
    
    # Test convolution
    results['convolution'] = test_neural_convolution(n_trials=10, dimensions=256)
    
    # Test encoding/decoding
    results['encoding_decoding'] = test_neural_encoding_decoding(n_trials=20, dimensions=256)
    
    # Test object queries
    results['object_query'] = test_neural_object_query(n_trials=10, dimensions=256)
    
    # Test location queries
    results['location_query'] = test_neural_location_query(n_trials=10, dimensions=256)
    
    # Summary
    print("=" * 60)
    print("SUMMARY OF NEURAL RESULTS")
    print("=" * 60)
    print(f"Convolution similarity:    {results['convolution']:.3f} (target: >0.85)")
    print(f"Encoding/Decoding:         {results['encoding_decoding']:.1f}% (target: 90-94%)")
    print(f"Object query:              {results['object_query']:.1f}% (target: 85-95%)")
    print(f"Location query:            {results['location_query']:.1f}% (target: 85-94%)")
    print("=" * 60)
    
    return results


if __name__ == '__main__':
    results = run_all_neural_tests()