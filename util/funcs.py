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


def _tokenize_expression(expr):
    """
    tokenizes an expression and returns the list
    :param expr: expression to be tokenized
    :return:
    """
    return re.findall(r'(?ms)\W*(\w+)', expr)


def _check_variable(string, varname):
    """
    check variable by using regular expressions
    :return:
    """
    prefix = '[(, ]'  # Any character with the exception of '(, '
    suffix = '[), ]'  # Any character with the exception of '), '
    pattern = prefix + varname + suffix
    re.match(pattern, string)
