"""
distrib.py
File containing all distributions and their parsing
"""
availableDists = ['dirac', 'normal', 'exponential', 'gamma', 'inverse_gamma', 'beta']


def CreateNode(distribution, parameters):
    if distribution is 'dirac':
        new_node = {
            'distr': 'Constant',
            'param': [None],
            'data': None
        }
    elif distribution is 'normal':
        new_node = {
            'distr': 'LogLikeGaussian',
            'param': [parameters['mu'], parameters['s2']],
            'data': None
        }
    elif distribution is 'exponential':
        new_node = {
            'distr': 'LogLikeExponential',
            'param': [parameters['lambda']],
            'data': None
        }
    elif distribution is 'gamma':
        new_node = {
            'distr': 'LogLikeGamma',
            'param': [parameters['alpha'], parameters['beta']],
            'data': None
        }
    elif distribution is 'inverse_gamma':
        new_node = {
            'distr': 'LogLikeInvGamma',
            'param': [parameters['alpha'], parameters['beta']],
            'data': None
        }
    elif distribution is 'beta':
        new_node = {
            'distr': 'LogLikeBeta',
            'param': [parameters['alpha'], parameters['beta']],
            'data': None
        }
    else:
        raise NameError('Unrecognized distribution: ' + distribution)

    return new_node
