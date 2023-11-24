import numpy as np
import matplotlib.pyplot as plt
import pymc as pm
import arviz as az

# generarea timpilor de asteptare
timp_asteptare = np.random.normal(loc=10, scale=2, size=200)

def solve():
    with pm.Model() as model:
        # distributiile a priori pentru miu si sigma
        priori_miu = pm.Normal('priori_miu', mu=10)
        priori_sigma = pm.Normal('priori_sigma')

        observation = pm.Normal("observation", mu=priori_miu, tau=priori_sigma,
                                observed=timp_asteptare)

    with model:
        trace = pm.sample(1000, tune=1000)
        az.plot_trace(trace)
        plt.show()

if __name__ == "__main__":
    solve()
