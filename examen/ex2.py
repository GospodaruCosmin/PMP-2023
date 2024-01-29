import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
import pandas as pd
import arviz as az
from scipy import stats


def posterior_grid(grid_points, heads):
    grid = np.linspace(0, 1, grid_points)
    prior = np.repeat(1 / grid_points, grid_points)
    likelihood = stats.binom.pmf(heads, heads, grid)
    posterior = likelihood * prior
    posterior /= posterior.sum()
    return grid, posterior


if __name__ == "__main__":
    # am aruncat de 13 ori si am obtinut 1 heads (deci 12 tails)
    data = np.repeat([0, 1], (12, 1))
    points = 12
    heads = data.sum()
    grid, posterior = posterior_grid(points, heads)
    plt.plot(grid, posterior, 'o-')
    plt.title(f'heads = {heads}')
    plt.yticks([])
    plt.xlabel('θ')
