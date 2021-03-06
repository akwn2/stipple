import matplotlib.pyplot as plt
from src.stipple import *
from time import time

if __name__ == '__main__':

    # True parameters
    N = 10
    w1 = 5.
    w2 = 1.

    # Generating inputs and outputs
    x = np.linspace(0, 10, N)
    theta = 1 / (1 + np.exp(-(w1 * x + w2)))
    y = np.random.rand(10, 1)

    for ii in xrange(0, N):
        if y[ii] >= theta[ii]:
            y[ii] = 1
        else:
            y[ii] = 0

    # Modelling in Stipple
    model = Stipple()

    tic = time()
    # Hyper-priors, non-informative
    model.input('x', x)
    model.assume('normal', 'w1', {'mu': 1., 's2': 10.})
    model.assume('normal', 'w2', {'mu': 5., 's2': 10.})
    model.assume('bernoulli', 'y', {'theta': 'Sigmoid(Add(Mult(w1, x), w2))'})
    model.observe('y', y)
    model.infer(method='HMC')

    print 'w1 mean = ' + str(model.results['w1']['mean'])
    print 'w1 var  = ' + str(model.results['w1']['var'])
    print 'w2 mean = ' + str(model.results['w2']['mean'])
    print 'w2 var  = ' + str(model.results['w2']['var'])

    toc = time()

    print('Total elapsed time: ' + str(toc - tic))
