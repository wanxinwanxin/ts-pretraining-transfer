# QUESTION

When we pretrain a model on a set of stochastic processes, and then try to forecast a series it has never seen, which properties of the pretraining distribution determine transfer?

We will be using a fixed architecture (not sure which one yet). Then we will vary properties of the pretraining distribution across arms. We will use regret against the Bayes-optimal forecaster as the measure. 

# HYPOTHESIS

We will define a set of operations below. Our hypothesis predicts that learning can only happen when the target distribution is within the closure of the pretraining distributions under the defined composition operations. Outside the closure, we expect regret to plateau above 0. We consider learning to happen when model performance approaches the Bayes-optimal forecaster.

If we identify a target distribution outside of the closure that we can achieve regret levels around 0 with enough data, then we reject the hypothesis.


# DISTRIBUTIONS
1. Autoregressive (AR/ARMA)
2. Seasonal
3. Trend + noise
4. Stochastic volatility
5. Regime-switching
6. Changepoint / piecewise trend
7. Threshold AR or other nonlinear (may serve as held-out family)


# PRETRAINING DATASET TABLE

Data 1-4 has the same volume (budget).


| DATA                      | Description                                                  |
| ------------------------- | ------------------------------------------------------------ |
| Narrow-A                  | AR + seasonal TS only                                  |
| Narrow-B                  | changepoint + regime-switching  TS only                          |
| Broad-balanced            | 6 distributions, equal volume                            |
| Broad-imbalanced          | 6 distributions, Pareto-like weights                            |
| Narrow-A @ 4× budget      | Narrow-A with increased volume (4x)                              |
| Curriculum (narrow→broad) | Train on narrow first, then broad. Tests importance of order |
| Oracle-matched            | Train = Target. Gives upper limit                            |


