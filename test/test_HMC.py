from unittest import TestCase
from util.hmc import *
import matplotlib.pyplot as plt


class TestHMC(TestCase):
    def test_hmc(self):
        # cov = np.array([[1, 0.998],  # Covariance example as in MacKay's book. Should be a line
        #                 [0.998, 1]])
        inv_cov = np.array([[+250.25, -249.75],
                            [-249.75, +250.25]])

        options = {
            'n_samples': 20,  # Number of samples
            'E': lambda x: 0.5 * np.dot(x.T, np.dot(inv_cov, x)),  # (Potential) energy function
            'dE': lambda x: np.dot(inv_cov, x),  # Derivative of the potential energy
            'N': 2,
            'max rejections': 5000,  # Maximum number of rejections
            'steps': 19,  # Number of St\"{o}rmer-Verlet steps
            'step size': 0.055,  # Step size to be used in the integration
            'max tune iter': 0,
        }

        samples = HMC(options).hmc()

        plt.plot(samples[:, 0], samples[:, 1], '-o')
        plt.show()

    def test_tune(self):
        self.fail()
