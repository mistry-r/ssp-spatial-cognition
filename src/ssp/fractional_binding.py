import numpy as np
from numpy.fft import fft, ifft

class SpatialSemanticPointer:
    """
    Implements Spatial Semantic Pointers using fractional binding.
    """

    def __init__(self, dimensions=512, seed=None):
        """
        Initialize SSP system with random unitary basis vectors.
        """
        if dimensions % 2 != 0:
            raise ValueError("Dimensions must be even for FFT operations.")
        
        self.d = dimensions
        self.rng = np.random.RandomState(seed)

        # Generate unitary basis vectors for X and Y axes
        self.X = self._generate_unitary_vector()
        self.Y = self._generate_unitary_vector()

    def _generate_unitary_vector(self):
        """
        Generates a random unitary vector.

        A unitary vector has the property that |B^k| = |B| for any k.
        This is achieved by having unit magnitude in the Fourier domain.
        """
        # Generate random phases in Fourier domain
        phases = self.rng.uniform(-np.pi, np.pi, self.d // 2 + 1)

        # Create complex values with unit magnitude (symmetry needed for real-valued output)
        fft_val = np.zeros(self.d, dtype=complex)
        fft_val[0] = 1.0 # DC component is real

        for i in range(1, self.d // 2):
            fft_val[i] = np.exp(1j * phases[i])
            fft_val[self.d - i] = np.conj(fft_val[i]) # Conjugate symmetry

        fft_val[self.d // 2] = np.exp(1j *phases[self.d // 2]) # Nyquist

        vec = ifft(fft_val).real # Convert to time domain
        vec = vec / np.linalg.norm(vec) # Normalize to unit length

        return vec
    
    def circular_convolution(self, a, b):
        """
        Compute circular convolution of two vectors 
        a ⊛ b = F^{-1}{F{a} ⊙ F{b}}
        """
        if len(a) != len(b):
            raise ValueError("Vectors must be of the same length for circular convolution.")
        
        result = ifft(fft(a) * fft(b)).real

        return result
    
    def get_inverse(self, vec):
        """
        Compute approximate inverse of a vector
        v ⊛ v^{-1} ≈ identity
        """
        inv = np.zeros_like(vec)
        inv[0] = vec[0] 
        inv[1:] = vec[-1:0:-1]

        return inv
    
    def fractional_power(self, base_vec, exponent):
        """
        Compute fractional power of a vector
        B^k = F^{-1}{F{B}^k}
        """
        fft_val = fft(base_vec) # Transform to Fourier domain
        fft_powered = np.power(fft_val, exponent)
        result = ifft(fft_powered).real # Transform back to time domain

        return result

    def encode_position(self, x, y):
        """
        Encode a 2D position (x, y) into an SSP
        S(x, y) = X^x ⊛ Y^y
        """
        X_to_x = self.fractional_power(self.X, x)
        Y_to_y = self.fractional_power(self.Y, y)

        ssp = self.circular_convolution(X_to_x, Y_to_y)

        ssp = ssp / np.linalg.norm(ssp)

        return ssp
    
    def encode_region(self, x_range, y_range, num_samples=100):
        """
        Encode a rectangular region defined by integrating over positions.
        S(R) = ∫_{(x,y)∈R} X^x ⊛ Y^y dx dy
        """
        x_min, x_max = x_range
        y_min, y_max = y_range

        x_samples = np.linspace(x_min, x_max, num_samples)
        y_samples = np.linspace(y_min, y_max, num_samples)

        region_ssp = np.zeros(self.d)

        for x in x_samples:
            for y in y_samples:
                region_ssp += self.encode_position(x, y)

        region_ssp = region_ssp / np.linalg.norm(region_ssp)

        return region_ssp
    
    def similarity(self, a, b):
        """
        Compute similarity between two vectors
        """
        return np.dot(a, b)

    def decode_position(self, ssp, bounds=(-5, 5), resolution=100):
        """
        Decode position from SSP by finding maximum similarity location
        """
        if isinstance(bounds, tuple):
            x_min, x_max = bounds
            y_min, y_max = bounds
        else:
            x_min, x_max = bounds[0]
            y_min, y_max = bounds[1]

        x_vals =  np.linspace(x_min, x_max, resolution)
        y_vals =  np.linspace(y_min, y_max, resolution)

        best_similarity = -np.inf
        best_pos = (0, 0)

        # Search for maximum similarity
        for x in x_vals:
            for y in y_vals:
                pos_ssp = self.encode_position(x, y)
                sim = self.similarity(ssp, pos_ssp)

                if sim > best_similarity:
                    best_similarity = sim
                    best_pos = (x, y)

        return best_pos

    def get_heatmap(self, ssp, bounds=(-5, 5), resolution=50):
        """
        Generate heatmap of similarities
        """
        if isinstance(bounds, tuple):
            x_min, x_max = bounds
            y_min, y_max = bounds
        else:
            x_min, x_max = bounds[0]
            y_min, y_max = bounds[1]

        x_vals =  np.linspace(x_min, x_max, resolution)
        y_vals =  np.linspace(y_min, y_max, resolution)

        X, Y = np.meshgrid(x_vals, y_vals)
        Z = np.zeros_like(X)

        for i, x in enumerate(x_vals):
            for j, y in enumerate(y_vals):
                pos_ssp = self.encode_position(x, y)
                Z[j, i] = self.similarity(ssp, pos_ssp)

        return X, Y, Z