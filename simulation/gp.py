import matplotlib.pyplot as plt
import numpy as np


class GPGenerator:
    """
    Generates samples from a Gaussian process with a given covariance matrix.
    """

    def __init__(self, kernel_function, structure: str = "additive", seed=None) -> None:
        np.random.seed(seed=seed)
        self.kernel_function = kernel_function
        self.structure = structure

    def generate_noisefree_sample(self, x: np.ndarray) -> np.ndarray:
        """
        Generate a sample from the Gaussian process with the given kernel
        function.
        """
        x = self._transform_x_to_matrix(x)
        return np.random.multivariate_normal(mean=np.zeros(x.shape[0]),
                                             cov=self.kernel_function(x, x))

    def generate_noisy_sample(self, x: np.ndarray, noise_variance: float) -> np.ndarray:
        """
        Generate a sample from the Gaussian process with the given kernel and noise.
        """
        return self.generate_noisefree_sample(x) + np.random.normal(scale=noise_variance, size=x.shape[0])

    def _transform_x_to_matrix(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        return x

    def plot_sample(self, x: np.ndarray, noise_variance: float = 0.0) -> None:
        """
        Plot a sample from the Gaussian process with the given kernel and noise.
        """
        y = self.generate_noisy_sample(x, noise_variance)
        if x.shape[1] == 1:
            if noise_variance > .0:
                plt.scatter(x, y)
            else:
                plt.plot(x, y)
            plt.show()
        elif x.shape[1] == 2:
            ax = plt.axes(projection='3d')
            if noise_variance == .0:
                # We draw a 3d surface plot of the GP
                ax.plot_trisurf(x[:, 0], x[:, 1],  y)
            plt.show()
