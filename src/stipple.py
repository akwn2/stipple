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
import util.distrib as distrib


class Stipple(object):
    def __init__(self):
        self._namespace = dict()
        self._stack = list()
        self._x = np.array([])
        self._dx = np.array([])
        self._input_list = list()
        self._output_list = list()
        self._var_list = list()
        self.results = dict()

        self._invalid_chars = ['#', '!', '@', '$']
        self._reserved_names = distrib.availableDists + adlib.op_list

    def __has_invalid_character(self, var_name):
        """
        checks if an invalid character is in var_name
        :param var_name:
        :return:
        """
        for item in self._invalid_chars:
            if item in var_name:
                return True

        return False

    def __is_reserved_name(self, var_name):
        """
        checks if the var_name is in the reserved names list
        :param var_name:
        :return:
        """
        for item in self._reserved_names:
            if item == var_name:
                return True

        return False

    def __is_already_defined(self, var_name):
        """
        checks if the var_name is in the reserved names list
        :param var_name:
        :return:
        """
        for item in self._namespace.keys():
            if item == var_name:
                return True

        return False

    def _check_variable_declaration(self, var_name):
        """
        checks if the variable declaration is valid
        :return:
        """
        if self.__has_invalid_character(var_name):
            raise NameError('Variable name contains invalid character: ' + var_name)
        if self.__is_reserved_name(var_name):
            raise NameError('Variable name reserved for operation:' + var_name)
        if self.__is_already_defined(var_name):
            raise NameError('Variable already defined: ' + var_name)

    def assume(self, distribution, name, parameters):
        """
        General assume statement for constructing the model
        :param distribution: string with the distribution type
        :param name: string with the symbol to be attributed to the distribution
        :param parameters: dictionary with the parameters of the distribution
        :return: error code if adding the node was unsuccessful
        """

        # INPUT CHECKING
        # Make sure people variable name follows conventions and is not already defined
        self._check_variable_declaration(name)

        # PARSING
        # Switch-case for available distributions in the language
        self._var_list.append(name)
        self._namespace[name] = distrib.CreateNode(distribution, parameters)

    def disregard(self, name):
        """
        removes a variable from the namespace and other internal control lists
        :param name: name of the variable to be removed
        :return:
        """
        del self._namespace[name]

        for ii in xrange(0, len(self._var_list)):
            if name == self._var_list[ii]:
                del self._var_list[ii]
                break

        for ii in xrange(0, len(self._input_list)):
            if name == self._input_list[ii]:
                del self._input_list[ii]
                break

        for ii in xrange(0, len(self._output_list)):
            if name == self._output_list[ii]:
                del self._output_list[ii]
                break

    def observe(self, name, data):
        """
        special assume statement with data recording
        :param name: name of the variable to be observed
        :param data: observed data points
        :return:
        """
        if name in self._namespace:  # variable needs to be declared beforehand
            self._output_list.append(name)
            self._namespace[name]['data'] = data
        else:
            raise NameError('Variable not declared: ' + name)

    def input(self, name, data):
        """
        special assume statement for input data
        :param name:
        :param data:
        :return:
        """
        # INPUT CHECKING
        # Make sure people variable name follows conventions and is not already defined
        self._check_variable_declaration(name)

        # PARSING
        # Switch-case for available distributions in the language
        self._namespace[name] = distrib.CreateNode('point', parameters=data, data=data)
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
                symbols = self._get_hmc_symbol_lookup()
                options = {
                    'n_samples': 100,
                    'E': lambda x: self._hmc_energy(x, expression, symbols),
                    'dE': lambda x: self._hmc_energy_grad(x, expression, symbols),
                    'N': len(self._var_list),
                    'max tune iter': 0,
                    'max rejections': 100
                }

                hmc_sampler = hmc.HMC(options=options)
                samples = hmc_sampler.run()

                for ii in xrange(0, len(self._var_list)):
                    name = self._var_list[ii]
                    self._namespace[name]['data'] = samples[:, ii]
                    self.results[name] = {
                        'mean': np.mean(samples[:, ii]),
                        'var': np.var(samples[:, ii])
                    }

            elif method == 'abc':
                raise NotImplementedError('')

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
        expression = ''
        suffix = ')'

        for idx in xrange(0, len(self._var_list)):

            var = self._var_list[idx]
            var_dist = self._namespace[var]['distr']

            arg_list = list()
            arg_list.append(var)

            # exclude cases where the variable symbol will be substituted for a constant
            if not (var_dist == 'Constant' or var_dist == 'Identity'):

                # convert parameters to strings
                for entry in self._namespace[var]['param']:
                    arg_list.append(str(entry))

                # Parse all arguments together
                args = ",".join(arg_list)

                if expression == '':
                    expression = 'Add(' + var_dist + '(' + args + '),'
                else:
                    expression += 'Add(' + var_dist + '(' + args + '),'
                    suffix += ')'

        return expression + '0.' + suffix  # fixme: this is a hack. substitute this

    def _get_hmc_symbol_lookup(self):
        """
        gets the symbol lookup dictionary for the hmc module
        :return:
        """
        # Create symbol lookup dictionary for adlib internal reference
        symbols = dict()
        for key in self._output_list:
            symbols[key] = self._namespace[key]['data']
        for key in self._input_list:
            symbols[key] = self._namespace[key]['data']

        # remove inputs and outputs from internal model structure
        ii = 0
        while ii < len(self._var_list):
            if (self._var_list[ii] in self._input_list) or (self._var_list[ii] in self._output_list):
                del self._var_list[ii]
            ii += 1

        return symbols

    def _hmc_energy(self, x, expression, symbols):
        """
        energy function of the hamiltonian monte-carlo
        :param x: input where the energy is evaluated at
        :param expression: expression string
        :param symbols: symbol dictionary for ADlib lookup
        :return:
        """
        # Add items to the dictionary locally
        for ii in xrange(0, len(self._var_list)):
            symbols[self._var_list[ii]] = x[ii]

        ad = adlib.ADLib(expression=expression, symbol_dict=symbols)
        return ad.eval()

    def _hmc_energy_grad(self, x, expression, symbols):
        """
        gradient of the energy function of the hamiltonian monte-carlo
        :param x: input where the gradient must be evaluated at
        :param expression: expression string
        :param symbols: symbol dictionary for ADlib lookup
        :return:
        """
        # Add items to the dictionary locally
        for ii in xrange(0, len(self._var_list)):
            symbols[self._var_list[ii]] = x[ii]

        # ad = adlib.ADLib(expression=expression, symbol_dict=symbols, grad_vars=self._var_list)
        ad = adlib.ADLib(expression=expression, symbol_dict=symbols)
        f_val, grad_dict = ad.eval(get_gradients=True)

        # Get values in the right order from the dictionary:
        dx = np.zeros_like(x)
        for ii in xrange(0, len(self._var_list)):
            dx[ii] = grad_dict[self._var_list[ii]]

        return dx
