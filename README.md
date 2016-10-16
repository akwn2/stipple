# stipple
Stipple is an acronym for "Simple Technical Interface for a Probabilistic Programming Engine". The motivation for such
name, beyond any aesthetics, is that the user has to be technically educated in probabilistic graphical models and the
allusion to sampling (the very first version of stipple had only simulation by sampling).

The idea behind Stipple is to implement a simple stack that allows the user to specify their model irrespective of the
inference procedure. The user can also add their own inference method using either distribution nodes (such as the ones
found in message-passing algorithms such as Expectation Propagation or Belief Propagation) as well as state and gradient
methods (as in Hamiltonian Monte Carlo and Variational Inference).

The user has to know how to specify generative models, and that is usually a reasonable assumption under all
probabilistic programming functions. A stipple model is defined by the following statements:
    * Assume: defines a variable in the model and which distribution it follows
    * Disregard: removes a variable from the model
    * Observe: adds data to a variable
    * Infer: finds the posterior or an approximation to the posterior of the model specified