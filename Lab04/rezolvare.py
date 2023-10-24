import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
import arviz as az

model = pm.Model()

with model:
    lambda_param = 20
    nr_clienti = pm.Poisson("Nr_clienti", mu=lambda_param, shape=1)
    comanda = pm.Normal("Comanda", mu=2, sigma=0.5)
    alpha = pm.Exponential("Gatit", lam=5)
    gatit = pm.Deterministic("Timp_Gatit", nr_clienti * alpha)
    timp_asteptare = comanda + gatit

    trace = pm.sample(500, chains=2, return_inferencedata=False)

timp_asteptare_mare_15 = np.sum(timp_asteptare > 15)
probabilitate = 1 - (timp_asteptare_mare_15 / len(trace))

print("Probabilitatea timpului total de așteptare <= 15: ", probabilitate)

timp_mediu_asteptare = np.mean(timp_asteptare)
print("Timpul mediu de așteptare: ", timp_mediu_asteptare)

az.plot_posterior(trace, var_names=["Nr_clienti", "Comanda", "Timp_Gatit"])
plt.show()
