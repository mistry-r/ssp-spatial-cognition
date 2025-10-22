import numpy as np
import matplotlib.pyplot as plt
from src.ssp.fractional_binding import SpatialSemanticPointer
from src.ssp.spatial_memory import SpatialMemory

def test_object_detection():
    """
    Replicate Figure 2 from Komer et al.
    Test single object, location, and missing object queries.
    """

    # Create SSP system
    ssp = SpatialSemanticPointer()
    memory = SpatialMemory(ssp)

    objects = {
        'FOX': (1.3, -1.2),
        'DOG': (1.1, 1.7),
        'BEAR': (2.4, -2.1),
        'WOLF': (-3.0, 2.5)
    }

    for obj_name, (x, y) in objects.items():
        memory.add_object(obj_name, x, y)

    # Test 1: Query object location 
    print("Test 1: Query single object")
    estimate_pos = memory.query_object('FOX')
    true_pos = objects['FOX']
    accuracy = np.linalg.norm(estimated_pos - true_pos) < 0.5
    print(f"Accuracy: {accuracy}")

    # Test 2: Query missing object
    print("\nTest 2: Query missing object")
    result = memory.query_object('BADGER')
    # TODO: Check similarity threshold

    # Test 3: Query location
    print("\nTest 3: Query location")
    detected_obj = memory.query_location(1.3, -1.2)
    print(f"Object at (1.3, -1.2): {detected_obj}")

    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    # TODO: Plot heatmaps

if __name__ == "__main__":
    test_object_detection()