import pandas as pd
import pymc3 as pm

data = pd.read_csv('Admission.csv')

GRE_scores = data['GRE']
GPA_scores = data['GPA']
admission_status = data['Admission']

if __name__ == "__main__":
    with pm.Model() as logistic_model:
        beta0 = pm.Normal('beta0', mu=0, sd=10)
        beta1 = pm.Normal('beta1', mu=0, sd=10)
        beta2 = pm.Normal('beta2', mu=0, sd=10)

        p = pm.invlogit(beta0 + beta1 * GRE_scores + beta2 * GPA_scores)

        observed = pm.Bernoulli('observed', p, observed=admission_status)

        step = pm.Metropolis()

        trace = pm.sample(10000, step=step)

    pm.summary(trace)
    pm.traceplot(trace)
