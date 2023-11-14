import pandas as pd
import matplotlib.pyplot as plt
import pymc as pm

df = pd.read_csv('auto-mpg.csv')
df = df[df['horsepower'] != '?']  # eliminam randurile ce contin '?'
df['horsepower'] = pd.to_numeric(df['horsepower'])

# ex. 2
with pm.Model() as model:
    alpha = pm.Normal('alpha', mu=0)
    beta = pm.Normal('beta', mu=0)

    mu = alpha + beta * df['horsepower']

    sigma = pm.Uniform('sigma', lower=0, upper=10)

    likelihood = pm.Normal('mpg', mu=mu, sigma=sigma, observed=df['mpg'])

if __name__ == '__main__':
    with model:
        trace = pm.sample(10, tune=10)
    for param in trace.posterior.variables:
        post_mean = trace.posterior[param].mean().item()
        post_std = trace.posterior[param].std().item()
        print(f"{param}: Mean = {post_mean:.2f}, SD = {post_std:.2f}")

    # ex. 3
    alpha_est = trace.posterior["alpha"].mean().item()
    beta_est = trace.posterior["beta"].mean().item()

    linear_regression = alpha_est + beta_est * df['horsepower']

    plt.figure(figsize=(10, 6))
    plt.scatter(df['horsepower'], df['mpg'], alpha=0.7)
    plt.plot(df['horsepower'], linear_regression, color='red')
    plt.title('Dreapta de regresie')
    plt.xlabel('Cai putere (CP)')
    plt.ylabel('Consum (mpg)')
    plt.grid(True)
    plt.show()
