"""
funcs.py
utility functions to be used throughout the stipple
"""
import re


# Utility functions for parsin
def str2num(s):
    """
    convert string to number
    """
    try:
        return int(s)
    except ValueError:
        return float(s)


def tokenize_expression(expr):
    """
    tokenizes an expression and returns the list
    :param expr: expression to be tokenized
    :return:
    """
    return re.findall(r"(?ms)\W*([\w\.]+)", expr)


def _check_variable(string, varname):
    """
    check variable by using regular expressions
    :return:
    """
    prefix = '[(, ]'  # Any character with the exception of '(, '
    suffix = '[), ]'  # Any character with the exception of '), '
    pattern = prefix + varname + suffix
    re.match(pattern, string)


def check_character(char, name, parameters):
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
