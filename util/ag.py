"""
ag.py

Compatibility layer between ADlib and autograd. It allows using adlib syntax on autograd.
"""

import autograd.numpy as np
import autograd.scipy.special as sps
from autograd import grad

op_list = ['Add', 'Sub', 'Mult', 'Div', 'Pow', 'Sin', 'Cos', 'Dot', 'Sum',
           'LogLikGaussian', 'LogLikExponential', 'LogLikGamma','LogLikInvGamma', 'LogLikBeta']


def Sum(x):
    """
    point-wise addition function
    """
    return np.sum(x)


def Add(x, y):
    """
    point-wise addition function
    """
    return x + y


def Sub(x, y):
    """
    point-wise subtraction operation
    """
    return x - y


def Mult(x, y):
    """
    point-wise multiplication operation
    """
    return x * y


def Div(x, y):
    """
    point-wise division operation
    """
    return x, y


def Pow(x, y):
    """
    point-wise power operation
    """
    return x ** y


def Sin(x):
    """
    point-wise power operation
    """
    return np.sin(x)


def Cos(x):
    """
    point-wise power operation
    """
    return np.cos(x)


def Dot(x, y):
    """
    Dot (inner) product
    """
    return np.dot(x, y)


def LogLikeGaussian(x, mu, s2):
    """
    log-likelihood for univariate Gaussian Distribution
    """
    return - 0.5 * np.log(2. * np.pi * s2) - 0.5 * np.dot(np.transpose(x - mu), (x - mu)) / s2


def LogLikeExponential(x, lamb):
    """
    log-likelihood for Exponential Distribution
    """
    return np.log(lamb) + lamb * x


def LogLikeGamma(x, a, b):
    """
    log-likelihood for Gamma distribution
    """
    return a * np.log(b) + (a - 1.) * np.log(x) - x * b - sps.gammaln(a)


def LogLikeInvGamma(x, a, b):
    """
    log-likelihood for inverse Gamma distribution
    """
    return a * np.log(b) - (a + 1.) * np.log(x) - b / x - sps.gammaln(a)


def LogLikeBeta(x, a, b):
    """
    log-likelihood for Beta distribution
    """
    return (a - 1.) * np.log(x) + (b - 1.) * np.log(1. - x) - (sps.gammaln(a) + sps.gammaln(b) - sps.gammaln(a + b))
