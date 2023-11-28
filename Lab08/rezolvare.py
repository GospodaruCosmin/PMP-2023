import pandas as pd
import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('Prices.csv')

price = data['Price'].values
speed = data['Speed'].values
hard_drive = np.log(data['HardDrive'].values)

if __name__ == "__main__":
    with pm.Model() as model:
        alpha = pm.Normal('alpha', mu=0, sd=10)
        beta1 = pm.Normal('beta1', mu=0, sd=10)
        beta2 = pm.Normal('beta2', mu=0, sd=10)
        sigma = pm.HalfNormal('sigma', sd=1)

        mu = alpha + beta1 * speed + beta2 * hard_drive

        likelihood = pm.Normal('likelihood', mu=mu, sd=sigma, observed=price)

        trace = pm.sample(20, tune=10)

    summary = pm.summary(trace)
    print(summary)
