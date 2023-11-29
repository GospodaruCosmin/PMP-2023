import pandas as pd
import pymc3 as pm
import numpy as np
import arviz as az
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

        mu = alpha + (beta1 * speed) + (beta2 * hard_drive)

        likelihood = pm.Normal('likelihood', mu=mu, sd=sigma, observed=price)

        trace = pm.sample(2000, tune=1000)

    az.plot_trace(trace)

    hdi_beta1 = az.hdi(trace['beta1'], hdi_prob=0.95)
    hdi_beta2 = az.hdi(trace['beta2'], hdi_prob=0.95)

    print(f"\n95% HDI pentru beta1: {hdi_beta1}")
    print(f"95% HDI pentru beta2: {hdi_beta2}")

    if hdi_beta1[0] > 0 and hdi_beta2[0] > 0:
        print("Frecventa procesorului si marimea hard diskului sunt predictori utili ai pretului de vanzare.")
    else:
        print("Frecventa procesorului si marimea hard diskului nu sunt predictori semnificativi ai pretului de vanzare.")

    simulated_samples = pm.sample_posterior_predictive(trace, samples=5000, model=model)
    simulated_prices = simulated_samples['likelihood']
    hdi_price = az.hdi(simulated_prices.flatten(), hdi_prob=0.90)
    print(f"\nInterval de 90% HDI pentru pretul așteptat: {hdi_price}")

    posterior_predictive = pm.sample_posterior_predictive(trace, samples=5000, model=model)
    simulated_prices_predictive = posterior_predictive['likelihood']
    hdi_prediction = az.hdi(simulated_prices_predictive.flatten(), hdi_prob=0.90)
    print(f"\nInterval de 90% HDI pentru pretul de vanzare prezis: {hdi_prediction}")
