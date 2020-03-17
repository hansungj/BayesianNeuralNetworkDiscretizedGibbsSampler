# BayesianNeuralNetworkDiscretizedGibbsSampler

**Bayesian Neural Network - Discretized Gibbs Sampler**

**Gibbs Sampler**
In order to tract the posterior parameter probability distribution, Gibbs sampler is used.
Since there is not a closed form solution for the conditional distribution necssary for the gibbs sampler P(w_i | w_1, w_2,..., w_i-1,w_i+1,...,w_k), it is not possible to properly sample from it. Hence discretized version is used. 
Posterior probability for each weight value in [-5,5] by increments of 0.1. This is possible by the Bayesian assumption for the parameters: true weight parameters will be centered around zero. 

**Prior Distribution**
Priors implemented in this project is "spike-and-slab" like mixture of normal distributions. Two normal distributions centered at zero with different scales is implemented. One normal component will have much smaller variance than the other allowing this component to be the “spike” part. This works as an automatic variable selection as it will set weights to zero unless backed by data. 
 
**Likelihood Function**
Likelihood function is a standard neural network regression model: a conditional distribution of y given the input x. This conditional distribution is a normal distribution.

**Burning Steps**
Gibbs sampler is to be run for ~300 iterations, but the first 40 models is discarded to allow proper mixing. (MCMC algorithm is guaranteed to converge for large number of steps n, but this n is usually not known) 



