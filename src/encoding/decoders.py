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
        
        Based on Lecture 3, Equation 7:
            D^T = (AA^T + Nσ²I)^{-1} A X^T
        """
        n_neurons, n_samples = activities.shape
        dimensions = targets.shape[0]
        
        # Compute A @ A^T
        AAT = activities @ activities.T
        
        # Add regularization: Nσ²I
        regularization = n_samples * (noise_sigma ** 2) * np.eye(n_neurons)
        
        # Solve: D^T = (AA^T + Nσ²I)^{-1} @ A @ X^T
        try:
            AAT_reg_inv = np.linalg.inv(AAT + regularization)
        except np.linalg.LinAlgError:
            # Use pseudo-inverse if singular
            AAT_reg_inv = np.linalg.pinv(AAT + regularization)
        
        decoders_T = AAT_reg_inv @ activities @ targets.T
        decoders = decoders_T.T
        
        return decoders
    
    def solve_function(self, activities, x_samples, function, noise_sigma=0.1):
        """
        Solve for function decoders.
        
        Based on Lecture 5: Feed-Forward Transformation
        """
        # Evaluate function at each sample
        n_samples = x_samples.shape[0]
        
        # Handle both scalar and vector outputs
        test_output = function(x_samples[0])
        if np.isscalar(test_output):
            output_dim = 1
            targets = np.array([function(x) for x in x_samples])
            targets = targets.reshape(1, -1)
        else:
            output_dim = len(test_output)
            targets = np.array([function(x) for x in x_samples]).T
        
        return self.solve(activities, targets, noise_sigma)