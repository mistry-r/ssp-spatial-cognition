import pytest
import numpy as np
from src.ssp.fractional_binding import SpatialSemanticPointer
from src.ssp.spatial_memory import SpatialMemory

class TestSpatialMemory:
    
    def test_initialization(self):
        """Test memory initialization."""
        ssp = SpatialSemanticPointer(dimensions=256, seed=42)
        memory = SpatialMemory(ssp, dimensions=256, seed=42)
        
        assert memory.d == 256
        assert len(memory.memory) == 256
        assert len(memory.vocabulary) == 0
        assert len(memory.objects) == 0
    
    def test_add_single_object(self):
        """Test adding a single object."""
        ssp = SpatialSemanticPointer(dimensions=256, seed=42)
        memory = SpatialMemory(ssp, dimensions=256, seed=42)
        
        memory.add_object('TEST', 1.0, 2.0)
        
        assert 'TEST' in memory.vocabulary
        assert 'TEST' in memory.objects
        assert len(memory.objects['TEST']) == 1
        assert memory.objects['TEST'][0] == (1.0, 2.0)
    
    def test_add_multiple_objects(self):
        """Test adding multiple objects."""
        ssp = SpatialSemanticPointer(dimensions=512, seed=42)
        memory = SpatialMemory(ssp, dimensions=512, seed=42)
        
        objects = {'A': (0, 0), 'B': (1, 1), 'C': (-1, -1)}
        
        for obj, (x, y) in objects.items():
            memory.add_object(obj, x, y)
        
        assert len(memory.vocabulary) == 3
        assert len(memory.objects) == 3
    
    def test_query_existing_object(self):
        """Test querying an object that exists."""
        ssp = SpatialSemanticPointer(dimensions=512, seed=42)
        memory = SpatialMemory(ssp, dimensions=512, seed=42)
        
        memory.add_object('TARGET', 2.0, -1.5)
        
        positions = memory.query_object('TARGET', bounds=(-5, 5), resolution=50)
        
        assert len(positions) > 0
        found_x, found_y = positions[0]
        error = np.sqrt((found_x - 2.0)**2 + (found_y - (-1.5))**2)
        assert error < 0.5
    
    def test_query_nonexistent_object(self):
        """Test querying an object that doesn't exist."""
        ssp = SpatialSemanticPointer(dimensions=512, seed=42)
        memory = SpatialMemory(ssp, dimensions=512, seed=42)
        
        memory.add_object('A', 0, 0)
        
        positions = memory.query_object('NONEXISTENT', bounds=(-5, 5))
        
        assert len(positions) == 0
    
    def test_query_location_with_object(self):
        """Test querying a location where an object exists."""
        ssp = SpatialSemanticPointer(dimensions=512, seed=42)
        memory = SpatialMemory(ssp, dimensions=512, seed=42)
        
        memory.add_object('ITEM', 1.5, -2.0)
        
        detected, similarity = memory.query_location(1.5, -2.0, threshold=0.1)
        
        assert detected == 'ITEM'
        assert similarity > 0.1
    
    def test_query_empty_location(self):
        """Test querying a location with no object."""
        ssp = SpatialSemanticPointer(dimensions=512, seed=42)
        memory = SpatialMemory(ssp, dimensions=512, seed=42)
        
        memory.add_object('ITEM', 3.0, 3.0)
        
        detected, similarity = memory.query_location(-3.0, -3.0, threshold=0.1)
        
        assert detected is None or similarity < 0.1
    
    def test_remove_object(self):
        """Test removing an object from memory."""
        ssp = SpatialSemanticPointer(dimensions=512, seed=42)
        memory = SpatialMemory(ssp, dimensions=512, seed=42)
        
        memory.add_object('TEMP', 0.0, 0.0)
        assert 'TEMP' in memory.objects
        
        memory.remove_object('TEMP', 0.0, 0.0)
        assert 'TEMP' not in memory.objects
    
    def test_duplicate_object_locations(self):
        """Test adding same object at multiple locations."""
        ssp = SpatialSemanticPointer(dimensions=512, seed=42)
        memory = SpatialMemory(ssp, dimensions=512, seed=42)
        
        memory.add_object('DUP', 1.0, 1.0)
        memory.add_object('DUP', -1.0, -1.0)
        
        assert len(memory.objects['DUP']) == 2
    
    def test_memory_unit_length(self):
        """Test that memory vector stays unit length."""
        ssp = SpatialSemanticPointer(dimensions=256, seed=42)
        memory = SpatialMemory(ssp, dimensions=256, seed=42)
        
        # Add several objects
        for i in range(5):
            memory.add_object(f'OBJ_{i}', float(i), float(i))
        
        # Memory should be unit length
        assert np.isclose(np.linalg.norm(memory.memory), 1.0, atol=1e-6)