import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
import pandas as pd
import arviz as az

if __name__ == "__main__":
    #citim fisierul
    data = pd.read_csv("BostonHousing.csv")

    # salvam coloanele de care avem nevoie
    rm = data['rm']
    crim = data['crim']
    indus = data['indus']

    # folosim distributii normale pentru fiecare dintre coloane
    with pm.Model() as model:
        rm_beta = pm.Normal('rm_beta', mu=0, sd=10)
        crim_beta = pm.Normal('crim_beta', mu=0, sd=10)
        indus_beta = pm.Normal('indus_beta', mu=0, sd=10)

        alpha = pm.Normal('alpha', mu=0, sd=10)

        # calculam mu necesar pentru a prezice variabila dependenta medv
        mu = alpha + rm_beta * rm + crim_beta * crim + indus_beta * indus

        medv = pm.Normal("medv", mu=mu, sd=10, observed=data["medv"])

        trace = pm.sample(1000, tune=1000)

        # pm.traceplot(trace)
        az.plot_trace(trace)
        plt.show()

        # obtinem estimarile de 95% pentru HDI ale parametrilor
        az.plot_forest(trace, hdi_prob=0.95)
        print(az.summary(trace, hdi_prob=0.95))

