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
import util.ag as ag
import util.hmc as hmc
import util.distrib as distrib


class Stipple(object):
    def __init__(self):
        self._stack = dict()
        self._namespace = dict()
        self._uid2name = dict()
        self._x = np.array([])
        self._dx = np.array([])

        self._uid_str = '__s_uid_'
        self._inputs = list()
        self._outputs = list()
        self._vars = list()

        self.results = dict()

        self._invalid_chars = ['#', '!', '@', '$']
        self.var_id_n = 0
        self._reserved_names = distrib.availableDists + ag.op_list

        self.options = {
            'AutoDiffEngine': 'autograd',  # can be one of the following: autograd/ADlib
            'InferMethod': 'HMC'
        }

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

        if self._uid_str in var_name:
            return True

        return False

    def __is_already_defined(self, var_name):
        """
        checks if the var_name is in the reserved names list
        :param var_name:
        :return:
        """
        for key in self._namespace.keys():
            if var_name == key:
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

        self.var_id_n += 1
        uid = self._uid_str + str(self.var_id_n)
        self._namespace[name] = uid
        self._uid2name[uid] = name
        self._vars.append(uid)

        # Now parse it into the appropriate inner variables and names
        for (par, par_xp) in parameters.items():
            par_xp = str(par_xp) # string cast here to prevent against raw numbers

            for (var_name, var_uid) in self._namespace.items():
                par_xp = par_xp.replace(var_name, var_uid)
            for operation in ag.op_list:
                par_xp = par_xp.replace(operation, 'ag.' + operation)

            parameters[par] = par_xp

        self._stack[uid] = distrib.CreateNode(distribution, parameters)

    def disregard(self, name):
        """
        removes a variable from the namespace and other internal control lists
        :param name: name of the variable to be removed
        :return:
        """
        unique_name = self._namespace.pop(name)

        del self._stack[unique_name]
        del self._uid2name[unique_name]

        for ii in xrange(0, len(self._vars)):
            if self._vars[ii] == unique_name:
                del self._vars[ii]
                break

        for ii in xrange(0, len(self._inputs)):
            if self._inputs[ii] == unique_name:
                del self._inputs[ii]
                break

        for ii in xrange(0, len(self._outputs)):
            if self._outputs[ii] == unique_name:
                del self._outputs[ii]
                break

    def observe(self, name, data):
        """
        special assume statement with data recording
        :param name: name of the variable to be observed
        :param data: observed data points
        :return:
        """
        if name in self._namespace.keys():  # variable needs to be declared beforehand
            uid = self._namespace[name]
            self._outputs.append(uid)   # add variable to outputs
            self._vars = [var for var in self._vars if var != uid]    # remove from outputs
            self._stack[uid]['data'] = data
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
        self.var_id_n += 1
        uid = self._uid_str + str(self.var_id_n)
        self._namespace[name] = uid
        self._uid2name[uid] = name
        self._inputs.append(uid)
        self._stack[uid] = distrib.CreateNode('point', parameters=data, data=data)

    def infer(self, method=None):
        """
        Infer statement for the constructed model
        :param method:
        :return:
        """
        if self._stack:
            # Method selector
            if not method:
                method = self.options['InferMethod']

            if method == 'HMC':

                if self.options['AutoDiffEngine'] == 'ADlib':
                    expression = self._parse_hmc_energy_adlib()
                    symbols = self._get_hmc_symbol_lookup()
                    options = {
                        'n_samples': 100,
                        'E': lambda x: self._hmc_energy(x, expression, symbols),
                        'dE': lambda x: self._hmc_energy_grad(x, expression, symbols),
                        'N': len(self._vars),
                        'max tune iter': 0,
                        'max rejections': 100
                    }

                elif self.options['AutoDiffEngine'] == 'autograd':

                    energy = self._parse_hmc_energy_ag()
                    grad_energy = ag.grad(energy)
                    options = {
                        'n_samples': 5000,
                        'E': energy,
                        'dE': grad_energy,
                        'N': len(self._vars),
                        'max tune iter': 0,
                        'max rejections': 5000
                    }

                else:
                    raise LookupError('Automatic differentiation engine not recognized: ' +
                                      self.options['AutoDiffEngine'])

                hmc_sampler = hmc.HMC(options=options)
                samples = hmc_sampler.run()

                for ii in xrange(0, len(self._vars)):
                    uid = self._vars[ii]
                    name = self._uid2name[uid]
                    self._stack[uid]['data'] = samples[:, ii]
                    self.results[name] = {
                        'mean': np.mean(samples[:, ii]),
                        'var': np.var(samples[:, ii])
                    }

            elif method == 'ABC':
                raise NotImplementedError('')

            elif method == 'VI':
                raise NotImplementedError('')

            elif method == 'EP':
                raise NotImplementedError('')

            else:
                raise EnvironmentError('Unrecognized inference method selected.')

        else:
            raise ReferenceError('No variable has been declared in the model.')

    def _parse_hmc_energy_ag(self):
        """
        parse for the energy function to be used in the hamiltonian monte-carlo using autograd
        :return:
        """
        expression = ''

        vars_and_outputs = self._vars + self._outputs
        for idx in xrange(0, len(vars_and_outputs)):

            var = vars_and_outputs[idx]
            var_dist = self._stack[var]['distr']

            arg_list = list()
            arg_list.append(var)

            # exclude cases where the variable symbol will be substituted for a constant
            if not (var_dist == 'Constant' or var_dist == 'Identity'):

                # convert parameters to strings
                for entry in self._stack[var]['param']:
                    arg_list.append(entry)

                # Parse all arguments together
                args = ",".join(arg_list)
                expression += '+ ag.' + var_dist + '(' + args + ')'

        # Now replace all variables
        idx = 0
        for uid in self._vars:
            if uid in expression:
                expression = expression.replace(uid, 'x[' + str(idx) + ']')
                idx += 1

        non_vars = self._inputs + self._outputs
        body = ""
        for uid in non_vars: # fixme: ideally no string conversion.
            try:
                body += "    " + uid + " = ag.np.array(" + str(np.ndarray.tolist(self._stack[uid]['data'])) + ")\n"
            except TypeError:
                # case we only got a float.
                body += "    " + uid + " = ag.np.array(" + str([self._stack[uid]['data']]) + ")\n"

        fun_str = "def energy(x):\n" + body + "    return ag.Sum(0. " + expression + ")"
        exec(fun_str)

        return energy

    def _parse_hmc_energy_adlib(self):
        """
        parse for the energy function to be used in the hamiltonian monte-carlo using adlib
        :return:
        """
        # fixme: this parsing scheme will result in an imbalanced tree. Divide it into a binary balanced tree.
        expression = ''
        suffix = ''

        for idx in xrange(0, len(self._vars)):

            var = self._vars[idx]
            var_dist = self._stack[var]['distr']
            var_data = self._stack[var]['data']

            arg_list = list()
            if var_data is None:
                arg_list.append(var)
            else:
                arg_list.append(var_data)

            # exclude cases where the variable symbol will be substituted for a constant
            if not (var_dist == 'Constant' or var_dist == 'Identity'):

                # convert parameters to strings
                for entry in self._stack[var]['param']:

                    for input_var in self._inputs:  # substitutes all variables for their actual values
                        if input_var in entry:
                            entry = entry.replace(input_var, self._stack[input_var]['data'])
                            break

                    arg_list.append(entry)

                # Parse all arguments together
                args = ",".join(arg_list)
                expression += 'Add(' + var_dist + '(' + args + '),'  # fixme: idea - each add statement can be parallel.
                suffix += ')'

        return expression + '0.' + suffix  # fixme: this is a hack. substitute this

    def _get_hmc_symbol_lookup(self):
        """
        gets the symbol lookup dictionary for the hmc module
        :return:
        """
        # Create symbol lookup dictionary for adlib internal reference
        symbols = dict()
        for key in self._outputs:
            symbols[key] = self._stack[key]['data']
        for key in self._inputs:
            symbols[key] = self._stack[key]['data']

        # remove inputs and outputs from internal model structure
        ii = 0
        while ii < len(self._vars):
            if (self._vars[ii] in self._inputs) or (self._vars[ii] in self._outputs):
                del self._vars[ii]
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
        for ii in xrange(0, len(self._vars)):
            symbols[self._vars[ii]] = x[ii]

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
        for ii in xrange(0, len(self._vars)):
            symbols[self._vars[ii]] = x[ii]

        # ad = adlib.ADLib(expression=expression, symbol_dict=symbols, grad_vars=self._var_list)
        ad = adlib.ADLib(expression=expression, symbol_dict=symbols)
        f_val, grad_dict = ad.eval(get_gradients=True)

        # Get values in the right order from the dictionary:
        dx = np.zeros_like(x)
        for ii in xrange(0, len(self._vars)):
            dx[ii] = grad_dict[self._vars[ii]]

        return dx
