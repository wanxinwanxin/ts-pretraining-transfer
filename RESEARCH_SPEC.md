# QUESTION

When we pretrain a model on a set of stochastic processes, and then try to forecast a series it has never seen, which properties of the pretraining distribution determine transfer?

We will be using a fixed architecture (not sure which one yet). Then we will vary properties of the pretraining distribution across arms. We will use regret against the Bayes-optimal forecaster as the measure. 

# HYPOTHESIS
We will define a set of operations below. Our hypothesis predicts that learning can only happen when the target distribution is within the closure of the pretraining distributions under the defined composition operations. Outside the closure, we expect regret to plateau above 0. We consider learning to happen when model performance approaches the Bayes-optimal forecaster.

If we identify a target distribution outside of the closure that we can achieve regret levels around 0 with enough data, then we reject the hypothesis.



