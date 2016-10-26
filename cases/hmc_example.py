import matplotlib.pyplot as plt
from src.stipple import *

if __name__ == '__main__':

    # True parameters
    N = 1
    w1 = 30.
    w2 = 20.
    s2 = 5.

    # Generating inputs and outputs
    x = np.linspace(0., 2. * np.pi, N)
    noise = s2 * np.random.randn(N)
    y = w1 * x + w2 * np.sin(x) + noise

    # Modelling in Stipple
    model = Stipple()

    # Hyper-priors, non-informative
    model.assume('normal', 'w1', {'mu': 15., 's2': 100.})
    model.assume('normal', 'w2', {'mu': 42., 's2': 100.})
    model.assume('inverse_gamma', 's2', {'alpha': 0.001, 'beta': 0.001})

    for idx in xrange(0, N):
        x_i = 'x_' + str(idx)
        y_i = 'y_' + str(idx)
        model.input(x_i, x[idx])
        model.assume('normal', y_i, {'mu': 'Add(Mult(w1,' + x_i + '), Mult(w1, Sin(' + x_i + ')) )', 's2': 's2'})
        model.observe(y_i, y[idx])

    model.infer(method='hmc')

    print 'w1 mean = ' + str(model.results['w1']['mean'])
    print 'w1 var  = ' + str(model.results['w1']['var'])
    print 'w2 mean = ' + str(model.results['w2']['mean'])
    print 'w2 var  = ' + str(model.results['w2']['var'])
    print 's2 mean = ' + str(model.results['s2']['mean'])
    print 's2 var  = ' + str(model.results['s2']['var'])

    print('Done!')
