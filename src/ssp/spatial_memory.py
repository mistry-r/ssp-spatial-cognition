import numpy as np

class SpatialMemory:
    """
    Represents a collection of objects at spatial locations.
    Implements Equation 7: M = Σ_i (OBJ_i ⊛ S_i)
    """

    def __init__(self, ssp_generator, dimensions=512, seed=None):
        self.ssp = ssp_generator
        self.d = dimensions
        self.rng = np.random.RandomState(seed)
        self.memory = np.zeros(dimensions) # Memory vector M
        self.vocabulary = {} # Vocabulary: object name -> semantic pointer
        self.objects = {} # Keep track of objects and their positions for analysis

    def _get_or_create_object_sp(self, object_name):
        """
        Get semantic pointer for an object, creating it if necessary.
        """
        if object_name not in self.vocabulary:
            # Create random semantic pointer
            sp = self.rng.randn(self.d)
            sp = sp / np.linalg.norm(sp)
            self.vocabulary[object_name] = sp

        return self.vocabulary[object_name]
    
    def add_object(self, object_name, x, y):
        """
        Add an object at position (x, y).
        M = M + (OBJ ⊛ S(x, y))
        """
        obj_sp = self._get_or_create_object_sp(object_name) # Get semantic pointer for object     
        pos_ssp = self.ssp.encode_position(x, y) # Encode position
        bound = self.ssp.circular_convolution(obj_sp, pos_ssp) # Bind object with position
        self.memory += bound # Add to memory
        
        # # Normalize memory
        # if np.linalg.norm(self.memory) > 0:
        #     self.memory = self.memory / np.linalg.norm(self.memory)
        
        # Track object
        if object_name not in self.objects:
            self.objects[object_name] = []
        self.objects[object_name].append((x, y))

    def remove_object(self, object_name, x, y):
        """
        Remove an object from memory.
        """
        if object_name not in self.vocabulary:
            return # Object not in memory
        
        obj_sp = self.vocabulary[object_name] # Get semantic pointer for object     
        pos_ssp = self.ssp.encode_position(x, y) # Encode position
        bound = self.ssp.circular_convolution(obj_sp, pos_ssp) # Bind object with position
        self.memory -= bound # Subtract from memory

        # # Normalize memory
        # if np.linalg.norm(self.memory) > 0:
        #     self.memory = self.memory / np.linalg.norm(self.memory)
        
        # Update tracking
        if object_name in self.objects:
            try:
                self.objects[object_name].remove((x, y))
                if len(self.objects[object_name]) == 0:
                    del self.objects[object_name]
            except ValueError:
                pass  # Position not found

    def normalize_memory(self):
        """
        Normalize the memory vector to unit length.
        """
        norm = np.linalg.norm(self.memory)
        if norm > 1e-10:
            self.memory = self.memory / norm
        
    def query_location(self, x, y, threshold=0.05):
        """
        Query what object is at position (x, y).
        Implements Equation 8: M ⊛ S(x,y)^{-1}
        """
        self.normalize_memory()

        pos_ssp = self.ssp.encode_position(x, y) # Encode position
        pos_inv = self.ssp.get_inverse(pos_ssp) # Get inverse of position SSP
        result = self.ssp.circular_convolution(self.memory, pos_inv) # Unbind memory with position inverse
        
        result_norm = np.linalg.norm(result)
        if result_norm > 1e-10:
            result = result / result_norm

        # Find most similar object in vocabulary
        similarities = {}
        for obj_name, obj_sp in self.vocabulary.items():
            sim = self.ssp.similarity(result, obj_sp)
            similarities[obj_name] = sim
        
        if len(similarities) == 0:
            return None, 0.0
        
        # Get best match
        best_object = max(similarities.keys(), key=lambda k: similarities[k])
        best_similarity = similarities[best_object]
        
        # Adaptive threshold based on vocabulary size
        n_items = len(self.vocabulary)
        adaptive_threshold = threshold * np.sqrt(n_items / 10.0)  # Scale with sqrt(n)
        adaptive_threshold = min(adaptive_threshold, 0.3)  # Cap at 0.3
        
        # Check if best is significantly better than second best (winner-take-all)
        sorted_sims = sorted(similarities.values(), reverse=True)
        if len(sorted_sims) > 1:
            margin = sorted_sims[0] - sorted_sims[1]
            # If margin is too small, might be noise
            if margin < 0.02 and best_similarity < 0.15:
                return None, best_similarity
        
        # Return only if above threshold
        if best_similarity > adaptive_threshold:
            return best_object, best_similarity
        else:
            return None, best_similarity

    def query_object(self, object_name, bounds=(-5, 5), resolution=100):
        """
        Query where an object is located.
        Implements Equation 9: M ⊛ OBJ^{-1}
        """
        if object_name not in self.vocabulary:
            return [] # Object not in memory
        
        self.normalize_memory()
        
        obj_sp = self.vocabulary[object_name].copy() # Get semantic pointer for object
        obj_sp = obj_sp / np.linalg.norm(obj_sp)
        
        obj_inv = self.ssp.get_inverse(obj_sp) # Get inverse of object SSP
        result_ssp = self.ssp.circular_convolution(self.memory, obj_inv) # Unbind memory with object inverse
        
        result_norm = np.linalg.norm(result_ssp)
        if result_norm > 1e-10:
            result_ssp = result_ssp / result_norm

        # Decode position(s); For single position, use decode_position -> ROHAN NOTE
        pos = self.ssp.decode_position(result_ssp, bounds, resolution=150)
        
        return [pos]
    
    def query_region(self, x_range, y_range, threshold=0.15):
        """
        Query which objects are in a spatial region.
        """
        self.normalize_memory()
        
        region_ssp = self.ssp.encode_region(x_range, y_range) # Encode region
        region_inv = self.ssp.get_inverse(region_ssp) # Get inverse
        result = self.ssp.circular_convolution(self.memory, region_inv) # Unbind from memory
        
        result_norm = np.linalg.norm(result)
        if result_norm > 1e-10:
            result = result / result_norm

        # Check all objects
        detected_objects = []
        for obj_name, obj_sp in self.vocabulary.items():
            sim = self.ssp.similarity(result, obj_sp)
            if sim > threshold:
                detected_objects.append((obj_name, sim))
        
        # Sort by similarity
        detected_objects.sort(key=lambda x: x[1], reverse=True)
        
        return detected_objects

    def get_heatmap(self, object_name=None, bounds=(-5, 5), resolution=50):
        """
        Generate heatmap for visualization
        """
        if object_name is not None:
            # Query specific object
            if object_name not in self.vocabulary:
                return None
            
            obj_sp = self.vocabulary[object_name]
            obj_inv = self.ssp.get_inverse(obj_sp)
            query_result = self.ssp.circular_convolution(self.memory, obj_inv)
            
            return self.ssp.get_heatmap(query_result, bounds, resolution)
        else:
            # Show overall memory
            return self.ssp.get_heatmap(self.memory, bounds, resolution)
        
    def get_memory_strength(self):
        """
        Get overall strength/magnitude of memory
        """
        return np.linalg.norm(self.memory)
    
    def clear(self):
        """
        Clear all memory contents
        """
        self.memory = np.zeros(self.d)
        self.objects = {}