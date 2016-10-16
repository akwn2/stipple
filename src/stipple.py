"""
# stipple
Stipple is an acronym for "SQL Tables Integrated with a Probabilistic Programming Engine". The motivation for such
name, beyond any aesthetics, is that the probabilistic graphical model interface can be integrated with and the
allusion to sampling (the very first version of stipple had only simulation by sampling).

The idea behind Stipple is to implement a simple stack that allows the user to specify their model irrespective of the
inference procedure. The user can also add their own inference method using either distribution nodes (such as the ones
found in message-passing algorithms such as Expectation Propagation or Belief Propagation) as well as state and gradient
methods (as in Hamiltonian Monte Carlo and Variational Inference).

The user has to know how to specify generative models, and that is usually a reasonable assumption under all
probabilistic programming functions. A stipple model is defined by the following statements:
    * Assume: defines a variable in the model and which distribution it follows
    * Disregard: removes a variable from the model
    * Observe: adds data to a variable
    * Infer: finds the posterior or an approximation to the posterior of the model specified
"""

import numpy as np
import util.adlib as adlib
import util.hmc as hmc


class Stipple(object):
    def __init__(self):
        self._namespace = dict()
        self._stack = list()
        self._x = np.array([])
        self._dx = np.array([])
        self._input_list = list()
        self._output_list = list()
        self._var_list = list()
        self.N = 0

    @staticmethod
    def _check_character(char, name, parameters):
        """
        checks if character is not in the name or parameter string (if it is, throw an error)
        :param char:
        :param name:
        :param parameters:
        :return:
        """
        if char in name:
            raise NameError('Invalid character in the variable name: ' + name)

        # Make sure people don't include # within the name of parameters
        for item in parameters.keys():
            if char in item:
                raise NameError('Invalid character in the variable parameters: ' + item)

    def assume(self, distribution, name, parameters):
        """
        General assume statement for constructing the model
        :param distribution: string with the distribution type
        :param name: string with the symbol to be attributed to the distribution
        :param parameters: dictionary with the parameters of the distribution
        :return: error code if adding the node was unsuccessfull
        """

        # INPUT CHECKING
        # Make sure people don't include # within the name of variables
        self._check_character('#', name, parameters)
        self._check_character('!', name, parameters)

        # Namespace checking to see if symbol already defined
        if name in self._namespace:
            raise NameError('Variable name already defined: ' + name)

        # PARSING
        # Switch-case for available distributions in the language
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

        self._var_list.append(name)
        self._namespace[name] = new_node

    def disregard(self, name):
        """
        removes a variable from the namespace and stack
        :param name: name of the variable to be removed
        :return:
        """
        # remove from namespace
        if not ('!' in name) and self.N > 0:
            self.N -= 1

        del self._namespace[name]
        # fixme: remove from variable, input and output lists

    def observe(self, name, data):
        """
        special assume statement with data recording
        :param name: name of the variable to be observed
        :param data: observed data points
        :return:
        """
        if name in self._namespace:
            self._namespace[name]['data'] = data
            self._output_list.append(name)
        else:
            raise NameError('Variable not declared: ' + name)

    def input(self, name, data):
        """
        special assume statement for input data
        :param name:
        :param data:
        :return:
        """
        self._check_character('#', name, {})
        self._check_character('!', name, {})

        # Namespace checking to see if symbol already defined
        if name in self._namespace:
            raise NameError('Variable name already defined: ' + name)

        # PARSING
        # Switch-case for available distributions in the language
        new_node = {
            'distr': 'Identity',
            'param': data,  # fixme: multiple inputs
            'data': data
        }
        self._namespace[name] = new_node
        self._input_list.append(name)

    def infer(self, method='hmc'):
        """
        Infer statement for the constructed model
        :param method:
        :return:
        """
        if self._namespace:
            # Method selector

            if method == 'hmc':
                expression = self._parse_hmc_energy()
                options = {
                    'n_samples': 5000,
                    'E': lambda x: self._hmc_energy(x, expression),
                    'dE': lambda x: self._hmc_energy_grad(x, expression),
                    'N': self.N,
                    'max tune iter': 0,
                    'max rejections': 5000
                }

                hmc_sampler = hmc.HMC(options=options)
                samples = hmc_sampler.run()
                return samples

            elif method == 'vi':
                raise NotImplementedError('')

            elif method == 'ep':
                raise NotImplementedError('')

            else:
                raise EnvironmentError('Unrecognized inference method selected.')

        else:
            raise ReferenceError('No variable has been declared in the model.')

    def _parse_hmc_energy(self):
        """
        parse for the energy function to be used in the hamiltonian monte-carlo
        :return:
        """
        # fixme: this parsing scheme will result in an imbalanced tree. Divide it into a binary balanced tree.
        # fixme: have to add a substitute for multiple data points
        expression = ''
        for idx in xrange(0, len(self._var_list)):

            var = self._var_list[idx]
            param = self._namespace[var]['param']

            arg_list = list()
            arg_list.append([var])

            max_length = 1
            if not param[0] is None:
                for entry in param:
                    if entry in self._input_list:
                        arg_list.append(self._namespace[entry]['data'])
                        if np.size(self._namespace[entry]['data']) > max_length:
                            max_length = np.size(self._namespace[entry]['data'])
                    else:
                        arg_list.append([str(entry)])

                if max_length > 1:
                    for ii in xrange(0, len(arg_list)):
                        if np.size(arg_list[0][ii]) == 1:
                            arg_list[0][ii] = arg_list[0][ii] * max_length
                            # fixme: what about the case where there are differing numbers of datapoints?

                # Transpose argument list
                arg_list = zip(*arg_list)
                for ii in xrange(0, max_length):
                    args = ",".join(arg_list[ii])
                    if expression == '':
                        expression = 'add(' + self._namespace[var]['distr'] + '(' + args + '),'
                        suffix = ')'
                    else:
                        expression += 'add(' + self._namespace[var]['distr'] + '(' + args + '),'
                        suffix += ')'

        return expression + '0.' + suffix  # fixme: this is a hack. substitute this

    def _hmc_energy(self, x, expression):
        """
        energy function of the hamiltonian monte-carlo
        :param x: input where the energy is evaluated at
        :param expression: expression string
        :return:
        """
        symbols = dict()
        for ii in xrange(0, len(self._var_list)):
            symbols[self._var_list[ii]] = str(x[ii])

        ad = adlib.ADLib(expression=expression, symbol_dict=symbols)
        return ad.eval()

    def _hmc_energy_grad(self, x, expression):
        """
        gradient of the energy function of the hamiltonian monte-carlo
        :param x: input where the gradient must be evaluated at
        :param expression: expression string
        :return:
        """
        symbols = dict()
        for ii in xrange(0, len(self._var_list)):
            symbols[self._var_list[ii]] = str(x[ii])
        ad = adlib.ADLib(expression=expression, symbol_dict=symbols)
        return ad.eval(get_gradients=True)
