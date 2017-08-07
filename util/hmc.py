"""
hmc.py
Implements the Hamiltonian Monte Carlo Inference algorithm
"""
import numpy as np
import util.linalg as la


# import pybo


class HMC:
    def __init__(self, options):
        """
        Class for Hamiltonian Monte Carlo
        :param options: option dictionary containing
        options['max rejections']   Maximum number of rejections
        options['max tune iter']    Maximum number of tuning iterations
        options['E']                Potential energy as a function of x only, i.e. dE(x)
        options['dE']               Derivative of the potential energy as a function of x only, i.e. dE(x)
        """
        # Unpack all parameters
        self.n_samples = options['n_samples']  # Number of samples
        self.potential = options['E']  # (Potential) energy function
        self.grad_potential = options['dE']  # Derivative of the potential energy
        self.n_states = options['N']

        if 'steps' in options.keys():  # Number of St\"{o}rmer-Verlet steps
            self.sv_steps = options['steps']
        else:
            self.sv_steps = 5  # Number of St\"{o}rmer-Verlet steps

        if 'step size' in options.keys():  # Number of St\"{o}rmer-Verlet steps
            self.step_size = options['step size']
        else:
            self.step_size = 0.0001  # Step size to be used in the integration

        if 'xin' in options.keys():  # Number of St\"{o}rmer-Verlet steps
            self.xin = options['xin']
        else:
            self.xin = np.ones([self.n_states, 1])

        # Tuner options
        self.max_tune_iter = options['max tune iter']
        self.inv_mat_m = np.eye(self.n_states)  # matrix m is always diagonal
        self.n_params = self.n_states + 2  # matrix_m + steps + delta
        self.max_reject = options['max rejections']  # Maximum number of rejections

    def hmc(self):
        """
        Standard Hamiltonian Monte Carlo implementation.
        :return: state samples
        """
        x0 = self.xin

        # Main iteration loop
        reject = 0
        accepted = 0
        samples = np.zeros([self.n_samples, self.n_states])
        x = x0
        while (accepted < self.n_samples) and (reject < self.max_reject):

            v = self.__momentum_sampler()  # Sample the momenta
            h_xv = self.__hamiltonian(x, v)  # Evaluate Hamiltonian

            for tt in range(0, self.sv_steps):  # Do leapfrog iterations
                v -= 0.5 * self.step_size * self.grad_potential(x)  # Half step in v
                x += 0.5 * self.step_size * self.__grad_kinetic(v)  # Full step in x
                v -= 0.5 * self.step_size * self.grad_potential(x)  # Half step in v

            h_new_xv = self.__hamiltonian(x, v)  # Get new value for Hamiltonian

            h_diff = h_new_xv - h_xv  # Analyse whether sample should be accepted
            if (not np.isnan(h_diff)) and ((h_diff < 0) or (np.log(np.random.rand(1, 1)) < -h_diff)):
                samples[accepted, :] = x.T
                x0 = x  # Update initial state
                accepted += 1
            else:
                x = x0  # Reset to initial state
                reject += 1

        if (reject >= self.max_reject) and (accepted < self.n_samples):
            print('!!! Warning: Maximum rejections reached. Total samples obtained: ' + str(accepted))
            return samples[0:accepted, :]
        else:
            print
            return samples

    def __hamiltonian(self, x, v):
        """
        Hamiltonian function of the system
        :param x:
        :param v:
        :return:
        """
        return 0.5 * la.qform(v, self.inv_mat_m, v) + self.potential(x)

    def __grad_kinetic(self, v):
        """
        Kinetic energy function gradient
        :param v: momentum
        :return:
        """
        return np.dot(self.inv_mat_m, v)

    def __momentum_sampler(self):
        """
        Samples a Gaussian momentum
        :return:
        """
        return np.dot(self.inv_mat_m, np.random.randn(self.inv_mat_m.shape[0], 1))

    def __hmc_wrapper(self, params):
        """
        Wrap MCMC for the bayesian optimisation tuning
        :param params:
        :return:
        """
        # Positive parameters for inv_mat_m positive definiteness
        self.inv_mat_m = np.diag(np.exp(params[0:self.n_states]))  # Kinetic energy matrix (mass matrix)
        self.sv_steps = np.floor(np.exp(params[self.n_states: self.n_states + 1]))  # Number of St\"{o}rmer-Verlet steps
        self.step_size = np.exp(params[-1])  # Step size to be used in the integration

        # Get samples
        samples = self.hmc()

        # Calculate effective sample size and return the minimum one (but could use another metric like average)
        min_ess = np.min(ess(samples))

        return min_ess

    def tune(self):
        """
        Tuning by Bayesian Optimisation
        :return:
        """
        # bounds = np.zeros([self.n_params, 2])
        # bounds[:, 1] = 1E6  # Right bound = +infty
        #
        # xbest, model, info = pybo.solve_bayesopt(objective=lambda x: self.__hmc_wrapper(x),
        #                                          bounds=bounds,
        #                                          niter=self.max_tune_iter,
        #                                          verbose=True)
        #
        # # Unpack learned parameters
        # xbest = np.exp(xbest)
        # self.inv_mat_m = np.diag(xbest[0:self.n_params])
        # self.sv_steps = np.floor(xbest[self.n_states: self.n_states + 1])
        # self.step_size = xbest[-1]

    def run(self):
        """
        Tunes and runs the HMC sampler
        :return:
        """
        if self.max_tune_iter > 0:
            self.tune()

        samples = self.hmc()
        return samples


# Auxiliary functions
def acf(x):
    """
    calculates the auto-correlation for a matrix of samples
    :param x: sample matrix with observations on its columns and output channels on its rows
    :return:
    """
    n_dim, n_obs = x.shape
    acorr = np.zeros([n_dim, 1])
    for dd in xrange(0, n_dim):
        acorr[dd] = np.correlate(x[dd, :], x[dd, :], mode='full')[x.shape[1] - 1]

    return acorr


def ess(x):
    """
    Calculates the effective sample size for a matrix of samples
    :param x: sample matrix with observations on its columns and output channels on its rows
    :return:
    """
    n_dim, n_obs = x.shape
    eta = n_obs * np.ones([n_dim, 1]) / (1 + (n_obs - 1) * acf(x))
    return eta
