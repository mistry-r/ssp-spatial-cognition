import numpy as np

class DecoderSolver:
    """
    Solves for optimal decoders using least-squares.
    
    Based on Lecture 3: Representation
    """
    
    def __init__(self):
        pass
    
    def solve(self, activities, targets, noise_sigma=0.1):
        """
        Solve for decoders using regularized least squares.
        
        Based on Lecture 3, Equation 7: D^T = (AA^T + Nσ²I)^{-1} A X^T
        
        Args:
            activities: Neural activities, shape (n_neurons, n_samples) or (n_samples, n_neurons)
            targets: Target values, shape (dimensions, n_samples) or (n_samples, dimensions)
            noise_sigma: Noise level for regularization
            
        Returns:
            Decoders with shape (n_neurons, dimensions)
        """
        activities = np.atleast_2d(activities)
        targets = np.atleast_2d(targets)
        
        # Determine which dimension is n_samples by finding matching dimension
        # activities could be (n_neurons, n_samples) or (n_samples, n_neurons)
        # targets could be (dimensions, n_samples) or (n_samples, dimensions)
        
        # Try all 4 orientations to find matching n_samples
        if activities.shape[1] == targets.shape[1]:
            # Both are (*, n_samples) - correct format
            n_samples = activities.shape[1]
        elif activities.shape[1] == targets.shape[0]:
            # activities is (n_neurons, n_samples), targets is (n_samples, dimensions)
            targets = targets.T
            n_samples = activities.shape[1]
        elif activities.shape[0] == targets.shape[1]:
            # activities is (n_samples, n_neurons), targets is (dimensions, n_samples)
            activities = activities.T
            n_samples = activities.shape[1]
        elif activities.shape[0] == targets.shape[0]:
            # Both are (n_samples, *)
            activities = activities.T
            targets = targets.T
            n_samples = activities.shape[1]
        else:
            raise ValueError(
                f"Cannot align activities {activities.shape} with targets {targets.shape}"
            )
        
        n_neurons = activities.shape[0]
        dimensions = targets.shape[0]
        
        # Sanity check
        assert activities.shape == (n_neurons, n_samples)
        assert targets.shape == (dimensions, n_samples)
        
        # Compute A @ A^T
        AAT = activities @ activities.T
        
        # Add regularization
        regularization = n_samples * (noise_sigma ** 2) * np.eye(n_neurons)
        
        # Solve: D^T = (AA^T + Nσ²I)^{-1} @ A @ X^T
        try:
            AAT_reg_inv = np.linalg.inv(AAT + regularization)
        except np.linalg.LinAlgError:
            AAT_reg_inv = np.linalg.pinv(AAT + regularization)
        
        # D^T shape: (n_neurons, dimensions)
        decoders_T = AAT_reg_inv @ activities @ targets.T
        
        return decoders_T
    
    def solve_function(self, activities, x_samples, function, noise_sigma=0.1):
        """
        Solve for function decoders.
        
        Args:
            activities: Neural activities (any orientation)
            x_samples: Input samples, shape (n_samples, input_dim)
            function: Function to compute on inputs
            noise_sigma: Noise level
            
        Returns:
            Decoders with shape (n_neurons, output_dim)
        """
        # Evaluate function for all samples
        targets_list = []
        for x in x_samples:
            y = function(x)
            y = np.atleast_1d(y)
            targets_list.append(y)
        
        targets = np.array(targets_list)  # Shape: (n_samples, output_dim)
        
        return self.solve(activities, targets, noise_sigma)