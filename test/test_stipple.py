from unittest import TestCase
from src.stipple import *


class TestStipple(TestCase):
    def test__parse_hmc_energy(self):
        # Parsing test without substituting for inputs and outputs
        model = Stipple()
        model.input('x', [0.9, 0.2])
        model.assume('dirac', 'c', {'lag': 0.1})
        model.assume('normal', 'f', {'mu': 0.2, 's2': 10.})
        expression = 'add(mult(c, x), f)'
        model.assume('normal', 'y', {'mu': expression, 's2': 1.0})
        xp = model._parse_hmc_energy()

        assert xp == 'add(LogLikeGaussian(f,0.2,10.0),add(LogLikeGaussian(y,add(mult(c, x), f),1.0),0.))'
