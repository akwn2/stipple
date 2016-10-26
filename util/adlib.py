"""
adlib.py
ADLib: Automatic Differentiation Library

This file contains the Automatic Differentiation functions to be used in Stipple.

Here we use the Forward Accumulation Method
"""
from funcs import *
import numpy as np
import scipy.special as sps

op_list = ['add', 'sub', 'mult', 'div', 'pow', 'sin', 'cos', 'dot',
           'Identity' 'LogLikGaussian', 'LogLikExponential', 'LogLikGamma',
           'LogLikInvGamma', 'LogLikBeta']


class ADLib:
    def __init__(self, expression, symbol_dict, grad_vars=None):
        """
        :param expression: expression to be parsed
        :param symbol_dict: list of symbols used for variables
        """
        self.expression = expression
        self.token_list = []

        self.symbol_dict = symbol_dict
        self.symbol_list = list(self.symbol_dict.keys())
        if grad_vars:
            self.grad_vars = grad_vars
        else:
            self.grad_vars = self.symbol_list

        self.op_list = ['Add', 'Sub', 'Mult', 'Div', 'Pow', 'Sin', 'Cos', 'Dot', 'Identity',
                        'LogLikeGaussian', 'LogLikeExponential', 'LogLikeGamma', 'LogLikeInvGamma', 'LogLikeBeta']

        # Check if the expression is valid, then tokenize it
        self.checker()
        self.token_list = tokenize_expression(self.expression)

    def checker(self):
        """
        checks if the expression supplied is allowed
        :return: None (raises an exception if
        """
        # Basic error-checking: parenthesis match
        left_par = self.expression.count('(')
        right_par = self.expression.count(')')

        if not (left_par == right_par):
            raise SyntaxError('!!! Error: Number of right and left parenthesis do not match')

        # Check if there are unrecognized operations
        checking = tokenize_expression(self.expression)

        allowed_tokens = self.op_list + self.symbol_list

        for token in checking:
            if not token in allowed_tokens:
                try:
                    if float(token):  # fixme this is a hack to check if it is a number
                        pass
                except:
                    raise SyntaxError('!!! Error: Unexpected token(s) in expression.')

    def eval(self, get_gradients=False):
        """
        parses an expression string into an evaluation tree
        :return: function for evaluation
        """

        stack = list()
        args_left = list()
        parent_node = None  # effectively is the same as the last-entry in the stack. Just used for readability

        uid_number = 0  # internal name for variables to avoid collision when computing gradients

        if get_gradients:
            grad_dict = dict((var, []) for var in self.grad_vars)

        while self.token_list:  # here we will build a n-ary tree with traversal using a stack (and not recursion)

            token = self.token_list.pop(0)  # get the next token in line

            # Evaluate token
            if token in self.op_list:  # Valid operation - In this case the node will have children
                node = eval(token + '()')
                stack.append(node)
                args_left.append(node.n_args)
                parent_node = node

            else:  # These nodes have no children and thus are leaf nodes.
                uid = '#uid' + str(uid_number)
                uid_number += 1

                if token in self.symbol_list:
                    # valid variable name, in which case we just substitute its value
                    node = Identity(self.symbol_dict[token])

                    if get_gradients:
                        # store unique ids to avoid collision problems and keep track to avoid searching when collecting
                        # the gradients
                        grad_dict[token].append(uid)
                        grad_dict[uid] = node.grad(node.arg_list)
                else:
                    # And the next point is also necessarily a constant
                    node = Constant(str2num(token))
                    if get_gradients:
                        grad_dict[uid] = node.grad(node.arg_list)

                if len(stack) == 0:
                    # This is the case when there is only a single entry which is a variable or a token
                    if get_gradients:
                        return node.func(node.arg_list), {token: node.grad(node.arg_list)}
                    else:
                        return node.func(node.arg_list)
                else:
                    # Add the computed item to the parent's argument list
                    arg_position = parent_node.n_args - args_left[-1]
                    parent_node.arg_list[arg_position] = node.func(node.arg_list)
                    args_left[-1] -= 1

                    if get_gradients:
                        parent_node.children[arg_position] = [uid]

            # Now collapse all possible operations
            # Note it may be more than one as you can chain multiple single-argument operations
            while args_left[-1] == 0:
                if len(stack) == 1:
                    # This is the last node with all arguments available, so we just have to evaluate it

                    if get_gradients:
                        node = stack[-1]
                        # Evaluate node's gradient and propagate its value to its children
                        grad_val = node.grad(node.arg_list)
                        for index in range(0, node.n_args):
                            for child_uid in node.children[index]:
                                try:
                                    grad_dict[child_uid][0] *= grad_val[index]
                                except ValueError:
                                    grad_dict[child_uid] *= grad_val[index]

                        # Collect all gradients and pass them onwards to the children
                        for var in self.symbol_list:
                            var_uids = grad_dict[var]
                            grad_dict[var] = 0
                            for uid in var_uids:
                                grad_dict[var] += grad_dict[uid][0]

                        return node.func(node.arg_list), grad_dict
                    else:
                        node = stack[-1]
                        return node.func(node.arg_list)

                else:
                    # Remove the last entries as we finished looking into them
                    args_left.pop()
                    stack.pop()

                    # As we completed all of the calculations for the argument of the parent node,
                    # treat it as a leaf node and evaluate it

                    node = parent_node  # make current parent a leaf
                    parent_node = stack[-1]  # up one level (get the parent of the current parent)
                    arg_position = parent_node.n_args - args_left[-1]
                    parent_node.arg_list[arg_position] = node.func(node.arg_list)  # compute its value
                    args_left[-1] -= 1

                    if get_gradients:
                        # Evaluate node's gradient and propagate its value to its children
                        grad_val = node.grad(node.arg_list)
                        for index in range(0, node.n_args):
                            for child_uid in node.children[index]:
                                try:
                                    grad_dict[child_uid][0] *= grad_val[index]
                                except ValueError:
                                    grad_dict[child_uid] *= grad_val[index]

                        # Flatten the the node's children list and pass it onwards to its parent
                        flat_children = [child for child_list in node.children for child in child_list]
                        parent_node.children[arg_position] = flat_children


# Function atoms
class FunctionNode:
    """
    prototype for node in the tree
    """

    def __init__(self):
        """
        constructor for the function node. Creates the properties:
        func: function
        grad: gradient
        n_args: number of arguments the function takes
        arg_list: list of arguments that the function takes
        """
        self.n_args = 0
        self.children = list()
        self.arg_list = list()


class Constant(FunctionNode):
    def __init__(self, x):
        FunctionNode.__init__(self)
        self.n_args = 0
        self.func = lambda y: x
        self.grad = lambda y: [np.zeros_like(x, dtype=np.float_)]


class Identity(FunctionNode):
    """
    constant function subclass
    """

    def __init__(self, x):
        FunctionNode.__init__(self)
        self.n_args = 0
        self.func = lambda y: x
        self.grad = lambda y: [np.ones_like(x, dtype=np.float_)]


class Add(FunctionNode):
    """
    point-wise addition function subclass
    """

    def __init__(self):
        FunctionNode.__init__(self)
        self.n_args = 2
        self.arg_list = [None] * self.n_args
        self.children = [[None]] * self.n_args

    @staticmethod
    def func(x):
        return x[0] + x[1]

    @staticmethod
    def grad(x):
        return [np.ones_like(x[0]), np.ones_like(x[1])]


class Sub(FunctionNode):
    """
    point-wise subtraction operation
    """

    def __init__(self):
        FunctionNode.__init__(self)
        self.n_args = 2
        self.arg_list = [None] * self.n_args
        self.children = [[None]] * self.n_args

    @staticmethod
    def func(x):
        return x[0] - x[1]

    @staticmethod
    def grad(x):
        return [np.ones_like(x[0]), -np.ones_like(x[1])]


class Mult(FunctionNode):
    """
    point-wise multiplication operation
    """

    def __init__(self):
        FunctionNode.__init__(self)
        self.n_args = 2
        self.arg_list = [None] * self.n_args
        self.children = [[None]] * self.n_args

    @staticmethod
    def func(x):
        return x[0] * x[1]

    @staticmethod
    def grad(x):
        return [x[1], x[0]]


class Div(FunctionNode):
    """
    point-wise division operation
    """

    def __init__(self):
        FunctionNode.__init__(self)
        self.n_args = 2
        self.arg_list = [None] * self.n_args
        self.children = [[None]] * self.n_args

    @staticmethod
    def func(x):
        return x[0] / x[1]

    @staticmethod
    def grad(x):
        return [1 / x[1], - x[0] / (x[1] ** 2)]


class Pow(FunctionNode):
    """
    point-wise power operation
    """

    def __init__(self):
        FunctionNode.__init__(self)
        self.n_args = 2
        self.arg_list = [None] * self.n_args
        self.children = [[None]] * self.n_args

    @staticmethod
    def func(x):
        return x[0] ** x[1]

    @staticmethod
    def grad(x):
        return [x[1] * x[0] ** (x[1] - 1), x[0] ** x[1] * np.log(x[0])]


class Sin(FunctionNode):
    """
    point-wise power operation
    """

    def __init__(self):
        FunctionNode.__init__(self)
        self.n_args = 1
        self.arg_list = [None] * self.n_args
        self.children = [[None]] * self.n_args

    @staticmethod
    def func(x):
        return np.sin(x)

    @staticmethod
    def grad(x):
        return [np.cos(x)]


class Cos(FunctionNode):
    """
    point-wise power operation
    """

    def __init__(self):
        FunctionNode.__init__(self)
        self.n_args = 1
        self.arg_list = [None] * self.n_args
        self.children = [[None]] * self.n_args

    @staticmethod
    def func(x):
        return np.cos(x)

    @staticmethod
    def grad(x):
        return [-np.sin(x)]


class Dot(FunctionNode):
    """
    Dot (inner) product
    """

    def __init__(self):
        FunctionNode.__init__(self)
        self.n_args = 2
        self.arg_list = [None] * self.n_args
        self.children = [[None]] * self.n_args

    @staticmethod
    def func(x):
        return np.dot(x[0], x[1])

    @staticmethod
    def grad(x):
        return [x[1].T, x[0].T]


class LogLikeGaussian(FunctionNode):
    """
    log-likelihood for univariate Gaussian Distribution
    """

    def __init__(self):
        FunctionNode.__init__(self)
        self.n_args = 3
        self.arg_list = [None] * self.n_args
        self.children = [[None]] * self.n_args

    @staticmethod
    def func(x):
        return - 0.5 * (np.log(2. * np.pi * x[2]) + (x[0] - x[1]) ** 2 / x[2])

    @staticmethod
    def grad(x):
        return [- (x[0] - x[1]) / x[2],
                + (x[0] - x[1]) / x[2],
                - 0.5 * (1 / x[2] + 1 / (x[2] ** 2) * (x[0] - x[1]) ** 2)]


class LogLikeExponential(FunctionNode):
    """
    log-likelihood for Exponential Distribution
    """

    def __init__(self):
        FunctionNode.__init__(self)
        self.n_args = 2
        self.arg_list = [None] * self.n_args
        self.children = [[None]] * self.n_args

    @staticmethod
    def func(x):
        return np.log(x[1]) + x[1] * x[0]

    @staticmethod
    def grad(x):
        return [x[1],
                1. / x[1] + x[0]]


class LogLikeGamma(FunctionNode):
    """
    log-likelihood for Gamma distribution
    """

    def __init__(self):
        FunctionNode.__init__(self)
        self.n_args = 3
        self.arg_list = [None] * self.n_args
        self.children = [[None]] * self.n_args

    @staticmethod
    def func(x):
        return x[1] * np.log(x[2]) + (x[1] - 1.) * np.log(x[0]) - x[0] * x[2] - sps.gammaln(x[1])

    @staticmethod
    def grad(x):
        return [(x[1] - 1.) / x[0] - x[2],
                np.log(x[2]) + np.log(x[0]) - sps.polygamma(0, x[1]),
                x[1] / x[2] - x[0]]


class LogLikeInvGamma(FunctionNode):
    """
    log-likelihood for inverse Gamma distribution
    """

    def __init__(self):
        FunctionNode.__init__(self)
        self.n_args = 3
        self.arg_list = [None] * self.n_args
        self.children = [[None]] * self.n_args

    @staticmethod
    def func(x):
        return x[1] * np.log(x[2]) - (x[1] + 1.) * np.log(x[0]) - x[2] / x[0] - sps.gammaln(x[1])

    @staticmethod
    def grad(x):
        return [- (x[1] + 1.) / x[0] + x[2] / (x[0] ** 2),
                np.log(x[2]) - np.log(x[1]) - sps.polygamma(0, x[1]),
                x[1] / x[2] - 1. / x[0]]


class LogLikeBeta(FunctionNode):
    """
    log-likelihood for Beta distribution
    """

    def __init__(self):
        FunctionNode.__init__(self)
        self.n_args = 3
        self.arg_list = [None] * self.n_args
        self.children = [[None]] * self.n_args

    @staticmethod
    def func(x):
        return (x[1] - 1.) * np.log(x[0]) + (x[2] - 1.) * np.log(1. - x[0]) - sps.betaln(x[1], x[2])

    @staticmethod
    def grad(x):
        return [(x[1] - 1.) / x[0] - (x[2] - 1.) / (1. - x[0]),
                np.log(x[0]) - sps.polygamma(0, x[1]) + sps.polygamma(0, x[1] + x[2]),
                np.log(1. - x[0]) - sps.polygamma(0, x[2]) + sps.polygamma(0, x[1] + x[2])]
