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
            activities: Neural activities - accepts any of:
                - (n_neurons, n_samples) 
                - (n_samples, n_neurons)
            targets: Target values - accepts any of:
                - (dimensions, n_samples)
                - (n_samples, dimensions)
            noise_sigma: Noise level for regularization
            
        Returns:
            Decoders with shape (n_neurons, dimensions)
        """
        # Ensure 2D arrays
        activities = np.atleast_2d(activities)
        targets = np.atleast_2d(targets)
        
        # Store original shapes for debugging
        orig_act_shape = activities.shape
        orig_targ_shape = targets.shape
        
        # Strategy: Figure out which dimension is n_samples by finding common dimension
        # n_samples should be the same in both activities and targets
        
        # Check all 4 possible interpretations:
        # 1. activities=(n_neurons, n_samples), targets=(dimensions, n_samples)
        # 2. activities=(n_neurons, n_samples), targets=(n_samples, dimensions)  
        # 3. activities=(n_samples, n_neurons), targets=(dimensions, n_samples)
        # 4. activities=(n_samples, n_neurons), targets=(n_samples, dimensions)
        
        if activities.shape[1] == targets.shape[1]:
            # Case 1: both are (*, n_samples) format - CORRECT FORMAT
            n_neurons = activities.shape[0]
            dimensions = targets.shape[0]
            n_samples = activities.shape[1]
        elif activities.shape[1] == targets.shape[0]:
            # Case 2: activities=(n_neurons, n_samples), targets=(n_samples, dimensions)
            # Need to transpose targets
            targets = targets.T
            n_neurons = activities.shape[0]
            dimensions = targets.shape[0]
            n_samples = activities.shape[1]
        elif activities.shape[0] == targets.shape[1]:
            # Case 3: activities=(n_samples, n_neurons), targets=(dimensions, n_samples)
            # Need to transpose activities
            activities = activities.T
            n_neurons = activities.shape[0]
            dimensions = targets.shape[0]
            n_samples = activities.shape[1]
        elif activities.shape[0] == targets.shape[0]:
            # Case 4: both are (n_samples, *) format
            # Need to transpose both
            activities = activities.T
            targets = targets.T
            n_neurons = activities.shape[0]
            dimensions = targets.shape[0]
            n_samples = activities.shape[1]
        else:
            # No common dimension - error
            raise ValueError(
                f"Cannot find common sample dimension. "
                f"Activities shape: {orig_act_shape}, Targets shape: {orig_targ_shape}"
            )
        
        # Final validation
        if activities.shape[1] != targets.shape[1]:
            raise ValueError(
                f"After reshaping, sample counts still don't match: "
                f"activities {activities.shape}, targets {targets.shape}"
            )
        
        # Now we have:
        # activities: (n_neurons, n_samples)
        # targets: (dimensions, n_samples)
        
        # Compute A @ A^T
        AAT = activities @ activities.T
        
        # Add regularization: Nσ²I
        regularization = n_samples * (noise_sigma ** 2) * np.eye(n_neurons)
        
        # Solve: D^T = (AA^T + Nσ²I)^{-1} @ A @ X^T
        try:
            AAT_reg_inv = np.linalg.inv(AAT + regularization)
        except np.linalg.LinAlgError:
            AAT_reg_inv = np.linalg.pinv(AAT + regularization)
        
        # D^T shape: (n_neurons, dimensions)
        decoders_T = AAT_reg_inv @ activities @ targets.T
        
        # Return D with shape (n_neurons, dimensions)
        return decoders_T
    
    def solve_function(self, activities, x_samples, function, noise_sigma=0.1):
        """
        Solve for function decoders
        
        Args:
            activities: Neural activities (any orientation)
            x_samples: Input samples, shape (n_samples, input_dim)
            function: Function to compute on inputs
            noise_sigma: Noise level
            
        Returns:
            Decoders with shape (n_neurons, output_dim)
        """
        n_samples = x_samples.shape[0]
        
        # Evaluate function at each sample
        test_output = function(x_samples[0])
        test_output = np.atleast_1d(test_output)
        output_dim = len(test_output)
        
        # Compute all outputs
        targets = np.array([np.atleast_1d(function(x)) for x in x_samples])
        
        # targets shape: (n_samples, output_dim)
        return self.solve(activities, targets, noise_sigma)