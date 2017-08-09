# stipple

Stipple is an acronym for "Simple Technical Interface for a Probabilistic Programming Engine".

This is a toy project to better understand the concepts behind probabilistic programming (i.e. implement reusable models that are detached from the inference procedure).

In general, proabilistic programming laguanges operate with the following statements:
  * Assume: defines a variable in the model and which distribution it follows
  * Disregard: removes a variable from the model
  * Observe: adds data to a variable
  * Infer: finds the posterior or an approximation to the posterior of the model specified

In this version, there are no optimisations, a stack for parsing the model and obtaining gradients through automatic differentiation (there are two versions, one is coded from scratch and the other uses an external library) and Hamiltonian Monte Carlo similar to what Stan does.

Hamiltonian Monte Carlo has no support for discrete variables, but one can extend that quite easily using continuous relaxations, see [this NIPS paper](https://papers.nips.cc/paper/4652-continuous-relaxations-for-discrete-hamiltonian-monte-carlo.pdf).)

Eventually I will implement other methods once I've got time free time to play with this project again.