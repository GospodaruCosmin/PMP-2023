import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt
import arviz as az


def main():
    az.style.use('arviz-darkgrid')
    dummy_data = np.loadtxt('dummy.csv')
    # pentru ex. 3, luam 500 de date aleatorii pentru x si y
    # dummy_data = np.random.rand(500, 2)

    x_1 = dummy_data[:, 0]
    y_1 = dummy_data[:, 1]
    order = 5
    x_1p = np.vstack([x_1 ** i for i in range(1, order + 1)])
    x_1s = ((x_1p - x_1p.mean(axis=1, keepdims=True)) /
            x_1p.std(axis=1, keepdims=True))
    y_1s = (y_1 - y_1.mean()) / y_1.std()

    sd_array = np.array([10, 0.1, 0.1, 0.1, 0.1])

    with pm.Model() as model_p:
        alpha = pm.Normal('alpha', mu=0, sd=10)
        beta = pm.Normal('beta', mu=0, sd=10, shape=order)
        # pentru ex1, subpunctul b, avem beta-urile de mai jos
        # beta = pm.Normal('beta', mu=0, sd=100, shape=order)
        # beta = pm.Normal('beta', mu=0, sd=sd_array, shape=order)
        epsilon = pm.HalfCauchy('epsilon', 5)

        mu = alpha + pm.math.dot(beta, x_1s)

        y_pred = pm.Normal('y_pred', mu=mu, sd=epsilon, observed=y_1s)

        trace_p = pm.sample(20, tune=10)

    pm.plot_posterior_predictive_glm(trace_p)

    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


if __name__ == "__main__":
    main()
