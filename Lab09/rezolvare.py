import pandas as pd
import pymc3 as pm
import matplotlib.pyplot as plt
import arviz as az

data = pd.read_csv('Admission.csv')

GRE_scores = data['GRE']
GPA_scores = data['GPA']
admission_status = data['Admission']

if __name__ == "__main__":
    with pm.Model() as model:
        beta0 = pm.Normal('beta0', mu=0, sd=10)
        beta1 = pm.Normal('beta1', mu=0, sd=10)
        beta2 = pm.Normal('beta2', mu=0, sd=10)

        p = pm.invlogit(beta0 + beta1 * GRE_scores + beta2 * GPA_scores)

        observed = pm.Bernoulli('observed', p, observed=admission_status)

        step = pm.Metropolis()

        trace = pm.sample(1000, step=step)


    pm.summary(trace)
    pm.traceplot(trace)

    new_GRE_score = 550
    new_GPA_score = 3.5
    new_GRE_score_2 = 500
    new_GPA_score_2 = 3.2

    new_logistic_probabilities = pm.invlogit(trace['beta0'] + trace['beta1'] * new_GRE_score + trace['beta2'] * new_GPA_score)
    new_logistic_probabilities_2 = pm.invlogit(trace['beta0'] + trace['beta1'] * new_GRE_score_2 + trace['beta2'] * new_GPA_score_2)

    hdi_new_student = az.hdi(new_logistic_probabilities, hdi_prob=0.9, show_intervals=True)
    hdi_new_student_2 = az.hdi(new_logistic_probabilities_2, hdi_prob=0.9, show_intervals=True)

    print("Intervalul de 90% HDI pentru probabilitatea ca studentul sa fie admis: ", hdi_new_student)
    print("Intervalul de 90% HDI pentru probabilitatea ca studentul 2 sa fie admis: ", hdi_new_student_2)

    # intervalele difera deoarece studentii au GPA-uri diferite, astfel daca majoritatea studentilor din baza de date
    # ar avea GPA-urile in jurul valorii primului student, atunci intervalul rezultat primului student este mai mic
    # (are sanse mai mari sa ofere predictia buna).
