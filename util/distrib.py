"""
distrib.py
File containing all distributions and their parsing
"""
availableDists = ['dirac', 'normal', 'exponential', 'gamma', 'inverse_gamma', 'beta']


def CreateNode(distribution, parameters, data=None):
    if distribution is 'dirac':  # This is an exceptional case, watch out
        new_node = {
            'distr': 'Constant',
            'param': [None],
            'data': parameters['lag']
        }
    elif distribution is 'point':  # This is an exceptional case, watch out
        new_node = {
            'distr': 'Identity',
            'param': parameters,
            'data': data
        }
    elif distribution is 'bernoulli':  # This is an exceptional case, watch out
        new_node = {
            'distr': 'LogLikeBernoulli',
            'param': [str(parameters['theta'])],
            'data': data
        }
    elif distribution is 'normal':
        new_node = {
            'distr': 'LogLikeGaussian',
            'param': [str(parameters['mu']), str(parameters['s2'])],
            'data': data
        }
    elif distribution is 'exponential':
        new_node = {
            'distr': 'LogLikeExponential',
            'param': [str(parameters['lambda'])],
            'data': data
        }
    elif distribution is 'gamma':
        new_node = {
            'distr': 'LogLikeGamma',
            'param': [str(parameters['alpha']), str(parameters['beta'])],
            'data': data
        }
    elif distribution is 'inverse_gamma':
        new_node = {
            'distr': 'LogLikeInvGamma',
            'param': [str(parameters['alpha']), str(parameters['beta'])],
            'data': data
        }
    elif distribution is 'beta':
        new_node = {
            'distr': 'LogLikeBeta',
            'param': [str(parameters['alpha']), str(parameters['beta'])],
            'data': data
        }
    else:
        raise NameError('Unrecognized distribution: ' + distribution)

    return new_node
