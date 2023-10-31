import pandas as pd
import pymc as pm
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('trafic.csv')
nr_masini = data['nr. masini'].values
num_minutes = len(nr_masini)

crestere_indices = [7, 16]
descrestere_indices = [8, 19]

intervals = [(4, 7), (7, 8), (8, 16), (16, 19), (19, 24)]
interval_capete = []

model = pm.Model()

with model:
    lambda_ = pm.Gamma('lambda', alpha=1, beta=0.1)
    traffic = pm.Poisson('traffic', mu=lambda_, value=nr_masini, observed=True)

    for i in crestere_indices:
        pm.Poisson(f'traffic_crestere_{i}', mu=traffic[i] * 1.2, value=nr_masini[i], observed=True)

    for i in descrestere_indices:
        pm.Poisson(f'traffic_descrestere_{i}', mu=traffic[i] * 0.8, value=nr_masini[i], observed=True)

model2 = pm.Model()
with model2:
    for i, interval in intervals:
        interval_data = nr_masini[(interval[0] * 60):(interval[1] * 60)]

        lambda_ = pm.Gamma('lambda', alpha=1, beta=0.1)
        traffic = pm.Poisson(f'traffic_{i}', mu=lambda_, observed=interval_data)

        interval_capete.append(interval)