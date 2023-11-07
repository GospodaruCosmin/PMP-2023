import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt
import arviz as az

Y = [0, 5, 10]
THETA = [0.2, 0.5]

with pm.Model() as model:
    n = pm.Poisson('n', mu=10)
    observed = []

    for y in Y:
        for theta in THETA:
            observed.append(pm.Binomial(f'observed_{y}_{theta}', n=n, p=theta, observed=y))

with model:
    trace = pm.sample(2000, tune=1000, chains=2)

az.plot_posterior(trace, var_names=['n'], ref_val=10)

plt.show()
