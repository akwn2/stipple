import matplotlib.pyplot as plt
from src.stipple import *
from time import time

if __name__ == '__main__':

    # True parameters
    N = 10
    w1 = 5.
    w2 = 1.
    s2 = 0.1

    # Generating inputs and outputs
    x = np.linspace(np.pi / 2., 3. * np.pi / 2, N)
    noise = s2 * np.random.randn(N)
    y = w1 * x + w2 * np.sin(x) + noise

    # Modelling in Stipple
    model = Stipple()

    tic = time()
    # Hyper-priors, non-informative
    model.input('x', x)
    model.assume('normal', 'w1', {'mu': 1., 's2': 10.})
    model.assume('normal', 'w2', {'mu': 5., 's2': 10.})
    model.assume('inverse_gamma', 's2', {'alpha': 0.01, 'beta': 0.01})
    model.assume('normal', 'y', {'mu': 'Add(Mult(w1, x), Mult(w2, Sin(x)) )', 's2': 's2'})
    model.observe('y', y)
    model.infer(method='HMC')

    print 'w1 mean = ' + str(model.results['w1']['mean'])
    print 'w1 var  = ' + str(model.results['w1']['var'])
    print 'w2 mean = ' + str(model.results['w2']['mean'])
    print 'w2 var  = ' + str(model.results['w2']['var'])

    toc = time()

    print('Total elapsed time: ' + str(toc - tic))
