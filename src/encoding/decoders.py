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
        """
        # Handle activities shape
        if activities.shape[0] > activities.shape[1]:
            # Likely (n_samples, n_neurons) - transpose
            activities = activities.T
        
        n_neurons, n_samples = activities.shape
        
        # Handle targets shape
        targets = np.atleast_2d(targets)
        if targets.shape[0] == n_samples:
            # Shape is (n_samples, dimensions) - transpose to (dimensions, n_samples)
            targets = targets.T
        
        dimensions = targets.shape[0]
        
        # Ensure same number of samples
        if activities.shape[1] != targets.shape[1]:
            raise ValueError(f"Mismatch: activities {activities.shape}, targets {targets.shape}")
        
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
        
        # We want D shape: (n_neurons, dimensions)
        # decoders_T already has the right shape!
        return decoders_T
    
    def solve_function(self, activities, x_samples, function, noise_sigma=0.1):
        """
        Solve for function decoders
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