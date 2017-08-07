from unittest import TestCase
from util.adlib import *
import numpy as np


class TestADlib(TestCase):
    def test_expression_checker_valid(self):
        expression = 'Mult(Add(x, 2), 4.5)'
        ADLib(expression, {'x': 1})

    def test_expression_checker_error(self):
        expression = 'Mult(Add(x, 2), # 4.5)'
        ADLib(expression, {'x': 1})

    def test_expression_eval_1(self):
        expression = 'Mult(Add(x, 2), 4.5)'
        parser = ADLib(expression, {'x': 1})
        result = parser.eval()
        np.testing.assert_allclose(13.5, result)

    def test_expression_eval_2(self):
        expression = 'Mult(4.5, Add(x, 2))'
        parser = ADLib(expression, {'x': 1})
        result = parser.eval()
        np.testing.assert_allclose(13.5, result)

    def test_expression_eval_3(self):
        expression = 'Sin(Mult(4.5, Add(x, 2)))'
        parser = ADLib(expression, {'x': 1})
        result = parser.eval()
        np.testing.assert_allclose(np.sin(13.5), result)

    def test_expression_eval_4(self):
        expression = 'Sin(Dot(x, y))'
        x = np.array([1, 2])
        y = np.array([3, 4])
        parser = ADLib(expression, {'x': x, 'y': y})
        result = parser.eval()
        np.testing.assert_allclose(np.sin(x.T.dot(y)), result)

    def test_expression_eval_5(self):
        expression = 'Add(LogLikeGaussian(x, 0.1, 0.5), LogLikeGaussian(y, 0.4, 0.9))'
        x = np.array([1])
        y = np.array([3])
        parser = ADLib(expression, {'x': x, 'y': y})
        result = parser.eval()
        np.testing.assert_allclose(-0.5 * np.log(2. * np.pi * 0.5) - 0.5 * (x - 0.1) ** 2 / 0.5
                                   - 0.5 * np.log(2. * np.pi * 0.9) - 0.5 * (y - 0.4) ** 2 / 0.9,
                                   result)

    def test_expression_grad_1(self):
        expression = 'Mult(4.5, Add(x, 2))'
        x = 2
        parser = ADLib(expression, {'x': x})
        result = parser.eval(get_gradients=True)
        ground_truth = 4.5
        np.testing.assert_allclose(ground_truth, result[1]['x'])

    def test_expression_grad_2(self):
        expression = 'Sin(Add(x, 2))'
        x = 2
        parser = ADLib(expression, {'x': x})
        result = parser.eval(get_gradients=True)
        ground_truth = np.cos(x + 2)
        np.testing.assert_allclose(ground_truth, result[1]['x'])

    def test_expression_grad_3(self):
        expression = 'Add( Div( Mult( 4.5, Add(x, 2) ), 3), 3)'
        x = 2
        parser = ADLib(expression, {'x': x})
        result = parser.eval(get_gradients=True)
        ground_truth = 4.5 / 3
        np.testing.assert_allclose(ground_truth, result[1]['x'])

    def test_expression_grad_4(self):
        expression = 'Sin( Mult( 4.5, Add(x, 2) ) )'
        x = 2
        parser = ADLib(expression, {'x': x})
        result = parser.eval(get_gradients=True)
        ground_truth = np.cos(4.5 * (x + 2)) * 4.5
        np.testing.assert_allclose(ground_truth, result[1]['x'])

    def test_expression_grad_5(self):
        expression = 'Add(Sin( Mult( 4.5, Add(x, 2) ) ), x)'
        x = 2
        parser = ADLib(expression, {'x': x})
        result = parser.eval(get_gradients=True)
        ground_truth = np.cos(4.5 * (x + 2)) * 4.5 + 1
        np.testing.assert_allclose(ground_truth, result[1]['x'])

    def test_expression_grad_6(self):
        expression = 'Mult(Sin( Mult( 4.5, Add(x, 2) ) ), x)'
        x = 2
        parser = ADLib(expression, {'x': x})
        result = parser.eval(get_gradients=True)
        ground_truth = np.cos(4.5 * (x + 2)) * 4.5 * x + np.sin(4.5 * (x + 2))
        np.testing.assert_allclose(ground_truth, result[1]['x'])

    def test_expression_grad_7(self):
        expression = 'Sin( Mult( 4.5, Dot(x, y) ) )'
        x = np.array([1, 2, 3])
        y = np.array([4, 5, 6])
        parser = ADLib(expression, {'x': x, 'y': y})
        result = parser.eval(get_gradients=True)
        ground_truth = np.cos(4.5 * x.dot(y)) * 4.5 * y
        np.testing.assert_allclose(ground_truth, result[1]['x'])

    def test_expression_grad_8(self):
        expression = 'Sin( Mult( 4.5, Dot(x, y) ) )'
        x = np.array([1, 2, 3])
        y = np.array([4, 5, 6])

        var_order = ['y', 'x']
        parser = ADLib(expression, {'x': x, 'y': y}, var_order)
        val, grad_dict = parser.eval(get_gradients=True)

        grad = list()
        for ii in xrange(0, len(var_order)):
            grad.append(grad_dict[var_order[ii]])

        grad = np.asarray(grad)
        gt = np.array([np.cos(4.5 * x.dot(y)) * 4.5 * x,
                       np.cos(4.5 * x.dot(y)) * 4.5 * y])

        np.testing.assert_allclose(gt, grad)


class TestIdentity(TestCase):
    def test_func(self):
        xin = 5
        f = Identity(xin)
        np.testing.assert_almost_equal(xin, f.func(f.arg_list))

    def test_grad(self):
        xin = 5
        f = Identity(xin)
        np.testing.assert_almost_equal(1., f.grad(f.arg_list))


class TestConstant(TestCase):
    def test_func(self):
        xin = 5
        f = Constant(xin)
        np.testing.assert_almost_equal(xin, f.func(f.arg_list))

    def test_grad(self):
        xin = 5
        f = Constant(xin)
        np.testing.assert_almost_equal(0., f.grad(f.arg_list))


class TestAdd(TestCase):
    def test_func(self):
        f = Add()
        f.arg_list = [5, 10]
        np.testing.assert_almost_equal(15, f.func(f.arg_list))

    def test_grad(self):
        f = Add()
        f.arg_list = [5, 10]
        np.testing.assert_almost_equal([1, 1], f.grad(f.arg_list))


class TestDot(TestCase):
    def test_func(self):
        f = Dot()
        a = np.array([1, 2])
        b = np.array([3, 4])
        f.arg_list = [a, b]
        np.testing.assert_almost_equal(a.dot(b), f.func(f.arg_list))

    def test_grad(self):
        f = Dot()
        a = np.array([1, 2])
        b = np.array([3, 4])
        f.arg_list = [a, b]
        np.testing.assert_almost_equal([b, a], f.grad(f.arg_list))
