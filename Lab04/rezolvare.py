import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
import arviz as az

model = pm.Model()

with model:
    lambda_param = 20
    nr_clienti = pm.Poisson("Nr_clienti", mu=lambda_param)
    comanda = pm.Normal("Comanda", mu=2, sigma=0.5)
    alpha = pm.Exponential("Gatit", lam=5)
    gatit = pm.Deterministic("Timp_Gatit", nr_clienti * alpha)
    timp_asteptare = comanda + gatit

