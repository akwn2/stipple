from unittest import TestCase
from src.stipple import *


class TestStipple(TestCase):
    def test__parse_hmc_energy(self):
        # Parsing test without substituting for inputs and outputs
        model = Stipple()
        model.input('x', [0.9, 0.2])
        model.assume('dirac', 'c', {'lag': 0.1})
        model.assume('normal', 'f', {'mu': 0.2, 's2': 10.})
        expression = 'Add(Mult(c, x), f)'
        model.assume('normal', 'y', {'mu': expression, 's2': 1.0})
        model.observe('y', np.array([0, 1]))
        xp = model._parse_hmc_energy()

        assert xp == 'Add(LogLikeGaussian(f,0.2,10.0),Add(LogLikeGaussian(y,Add(Mult(c, x), f),1.0),0.))'

    def test__hmc_energy(self):
        # Parsing test without substituting for inputs and outputs
        model = Stipple()
        x = np.array([0.9])
        c = 0.1
        mu_f = 0.2
        s2_f = 10.
        s2_y = 1.0
        mu_g = 0.3
        s2_g = 5.
        y = np.array([0.5])

        def energy_gt(xin):
            f = xin[0]
            g = xin[1]
            prior = (- 0.5 * np.log(2 * np.pi * s2_f) - 0.5 * (f - mu_f) ** 2 / s2_f
                     - 0.5 * np.log(2 * np.pi * s2_g) - 0.5 * (g - mu_g) ** 2 / s2_g)
            like = - 0.5 * np.log(2 * np.pi * s2_y) - 0.5 * (y - (g * c * x + f)) ** 2 / s2_y

            return prior + like

        # Model
        model.input('x', x)
        model.input('c', c)
        model.assume('normal', 'f', {'mu': mu_f, 's2': s2_f})
        model.assume('normal', 'g', {'mu': mu_g, 's2': s2_g})
        model.assume('normal', 'y', {'mu': 'Add(Mult(g, Mult(c, x)), f)', 's2': s2_y})
        model.observe('y', y)
        expr = model._parse_hmc_energy()
        symbols = model._get_hmc_symbol_lookup()

        def energy_calc(xin):
            return model._hmc_energy(xin, expr, symbols)

        f_in = np.array([0.1, 0.8])
        calc_val = energy_calc(f_in)
        gt_val = energy_gt(f_in)
        np.testing.assert_almost_equal(gt_val, calc_val)

    def test__hmc_energy_grad(self):
        # Parsing test without substituting for inputs and outputs
        model = Stipple()
        x = np.array([0.9])
        c = 0.1

        mu_f = 0.2
        s2_f = 10.
        mu_g = 0.3
        s2_g = 5.

        s2_y = 1.0
        y = np.array([0.5])

        def grad_energy_gt(xin):
            f = xin[0]
            g = xin[1]

            grad_f = - (f - mu_f) / s2_f + (y - (g * c * x + f)) / s2_y
            grad_g = - (g - mu_g) / s2_g + (y - (g * c * x + f)) / s2_y * c * x

            return np.array([grad_f[0], grad_g[0]])

        # Model
        model.input('x', x)
        model.input('c', c)
        model.assume('normal', 'f', {'mu': mu_f, 's2': s2_f})
        model.assume('normal', 'g', {'mu': mu_g, 's2': s2_g})
        model.assume('normal', 'y', {'mu': 'Add(Mult(g, Mult(c, x)), f)', 's2': s2_y})
        model.observe('y', y)
        expr = model._parse_hmc_energy()
        symbols = model._get_hmc_symbol_lookup()

        def grad_energy_calc(xin):
            return model._hmc_energy_grad(xin, expr, symbols)

        x_0 = np.array([0.1, 0.2])
        grad_calc_val = grad_energy_calc(x_0)
        grad_gt_val = grad_energy_gt(x_0)
        np.testing.assert_almost_equal(grad_gt_val, grad_calc_val)

    def test__gaussian_example(self):
        # inputs
        x = np.array([0.9])
        c = 0.1
        mu_f = 0.2
        s2_f = 10.
        mu_g = 0.3
        s2_g = 5.
        s2_y = 1.0
        y = np.array([0.0])

        # modelling
        model = Stipple()
        model.input('x', x)
        model.input('c', c)
        model.assume('normal', 'f', {'mu': mu_f, 's2': s2_f})
        model.assume('normal', 'y', {'mu': 'Add(f, c)', 's2': s2_y})
        model.observe('y', y)
        samples = model.infer()

        print('Done!')
