from unittest import TestCase
from util.adlib import *
import numpy as np


class TestADlib(TestCase):
    def test_expression_checker_valid(self):
        expression = 'mult(add(x, 2), 4.5)'
        ADLib(expression, {'x': 1})

    def test_expression_checker_error(self):
        expression = 'mult(add(x, 2), # 4.5)'
        ADLib(expression, {'x': 1})

    def test_expression_eval_1(self):
        expression = 'mult(add(x, 2), 4.5)'
        parser = ADLib(expression, {'x': 1})
        result = parser.eval()
        np.testing.assert_allclose(13.5, result)

    def test_expression_eval_2(self):
        expression = 'mult(4.5, add(x, 2))'
        parser = ADLib(expression, {'x': 1})
        result = parser.eval()
        np.testing.assert_allclose(13.5, result)

    def test_expression_eval_3(self):
        expression = 'sin(mult(4.5, add(x, 2)))'
        parser = ADLib(expression, {'x': 1})
        result = parser.eval()
        np.testing.assert_allclose(np.sin(13.5), result)

    def test_expression_grad_1(self):
        expression = 'mult(4.5, add(x, 2))'
        x = 2
        parser = ADLib(expression, {'x': x})
        result = parser.eval(get_gradients=True)
        ground_truth = 4.5
        np.testing.assert_allclose(ground_truth, result[1]['x'])

    def test_expression_grad_2(self):
        expression = 'sin(add(x, 2))'
        x = 2
        parser = ADLib(expression, {'x': x})
        result = parser.eval(get_gradients=True)
        ground_truth = np.cos(x + 2)
        np.testing.assert_allclose(ground_truth, result[1]['x'])

    def test_expression_grad_3(self):
        expression = 'add( div( mult( 4.5, add(x, 2) ), 3), 3)'
        x = 2
        parser = ADLib(expression, {'x': x})
        result = parser.eval(get_gradients=True)
        ground_truth = 4.5 / 3
        np.testing.assert_allclose(ground_truth, result[1]['x'])

    def test_expression_grad_4(self):
        expression = 'sin( mult( 4.5, add(x, 2) ) )'
        x = 2
        parser = ADLib(expression, {'x': x})
        result = parser.eval(get_gradients=True)
        ground_truth = np.cos(4.5 * (x + 2)) * 4.5
        np.testing.assert_allclose(ground_truth, result[1]['x'])

    def test_expression_grad_5(self):
        expression = 'add(sin( mult( 4.5, add(x, 2) ) ), x)'
        x = 2
        parser = ADLib(expression, {'x': x})
        result = parser.eval(get_gradients=True)
        ground_truth = np.cos(4.5 * (x + 2)) * 4.5 + 1
        np.testing.assert_allclose(ground_truth, result[1]['x'])

    def test_expression_grad_6(self):
        expression = 'mult(sin( mult( 4.5, add(x, 2) ) ), x)'
        x = 2
        parser = ADLib(expression, {'x': x})
        result = parser.eval(get_gradients=True)
        ground_truth = np.cos(4.5 * (x + 2)) * 4.5 * x + np.sin(4.5 * (x + 2))
        np.testing.assert_allclose(ground_truth, result[1]['x'])


class TestIdentity(TestCase):
    def test_func(self):
        xin = 5
        a = Identity(xin)
        np.testing.assert_almost_equal(xin, a.func(a.arg_list))

    def test_grad(self):
        xin = 5
        a = Identity(xin)
        np.testing.assert_almost_equal(1., a.grad(a.arg_list))


class TestConstant(TestCase):
    def test_func(self):
        xin = 5
        a = Constant(xin)
        np.testing.assert_almost_equal(xin, a.func(a.arg_list))

    def test_grad(self):
        xin = 5
        a = Constant(xin)
        np.testing.assert_almost_equal(0., a.grad(a.arg_list))


class TestAdd(TestCase):
    def test_func(self):
        a = Add()
        a.arg_list = [5, 10]
        np.testing.assert_almost_equal(15, a.func(a.arg_list))

    def test_grad(self):
        a = Add()
        a.arg_list = [5, 10]
        np.testing.assert_almost_equal([1, 1], a.grad(a.arg_list))
