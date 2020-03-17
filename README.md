# BayesianNeuralNetworkDiscretizedGibbsSampler

**Bayesian Neural Network - Discretized Gibbs Sampler**

**Gibbs Sampler**
In order to tract the posterior parameter probability distribution, Gibbs sampler is used.
Since there is not a closed form solution for the conditional distribution necssary for the gibbs sampler P(w_i | w_1, w_2,..., w_i-1,w_i+1,...,w_k), it is not possible to properly sample from it. Hence discretized version is used. Posterior probability of each weight value in [-5,5] by increments of 0.1 is calculated. The final weight value is then sampled from these probabilities. This is possible by the Bayesian assumption for the parameters: true weight parameters will be around zero. 

**Prior Distribution**
Priors implemented in this project is "spike-and-slab" like mixture of normal distributions. Two normal distributions centered at zero with different scales is implemented. One normal component will have much smaller variance than the other allowing this component to be the “spike” part. This works as an automatic variable selection as it will set weights to zero unless backed by data. 
 
**Likelihood Function**
Likelihood function is a standard neural network regression model: a conditional distribution of target t given the input x P(t|y(x,w)). This conditional distribution is a normal distribution.

**Burning Steps**
Gibbs sampler is to be run for ~300 iterations, but the first 40 models is discarded to allow proper mixing. (Metropolis-Hasting MCMC algorithm is guaranteed to converge for large number of steps n, but this n is usually not known) 

**References**
1. Xiong - Bayesian prediction of tissue-regulated splicing using RNA sequence and cellular context https://academic.oup.com/bioinformatics/article/27/18/2554/182135
2. Bishop - Pattern Recognition and Machine Learning https://cds.cern.ch/record/998831/files/9780387310732_TOC.pdf?source=post
3. Neal - BAYESIAN LEARNING FOR NEURAL NETWORKS
4. Blundell - Weight Uncertainty in Neural Network https://arxiv.org/pdf/1505.05424.pdf
