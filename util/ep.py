"""
ep.py
Parses an expression for and evaluates the Expectation Propagation algorithm
"""
import numpy as np
import util.linalg as la


class EP:
    def __init__(self):
        pass

    def _parse(self):
        """
        parses a graphical model into EP format
        :return:
        """
        pass

    def loop(self):
        pass


class EPnode:
    def __init__(self):
        self.n_par = 0
        self.param = list()
        self.uid = None
        self.variables = list()
        self.neighbours = list()


class GaussianNode(EPnode):
    def __init__(self):
        EPnode.__init__(self)
        self.n_par = 3
        self.param = list()
