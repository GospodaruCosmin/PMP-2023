from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD

problema = BayesianNetwork([('Cutremur', 'Incendiu'), ('Cutremur', 'Alarma'), ('Incendiu', 'Alarma')])

incendiu_cpd = TabularCPD(variable='Incendiu', variable_card=2, values=[[0.99, 0.01], [0.97, 0.03]], evidence=['Cutremur'], evidence_card=[2])

cutremur_cpd = TabularCPD(variable='Cutremur', variable_card=2, values=[[0.9995], [0.0005]])

alarma_cpd = TabularCPD(variable='Alarma', variable_card=2, values=[[0.9999, 0.0001, 0.98, 0.02], [0.8, 0.2, 0.02, 0.98]], 
                        evidence=['Cutremur', 'Incendiu'], evidence_card=[2, 2])

problema.add_cpds(incendiu_cpd, cutremur_cpd, alarma_cpd)

problema.check_model
print(problema) 
if(problema.check_model):
    print(True)
else:
    print(False)

# 2
p_alarma_date_cutremur = 0.02
p_cutremur = 0.0005
p_not_cutremur = 1 - p_cutremur
p_alarma_fara_cutremur = 0.0001

p_alarma = (p_alarma_date_cutremur * p_cutremur) + (p_alarma_fara_cutremur * p_not_cutremur)

p_cutremur_date_alarma = (p_alarma_date_cutremur * p_cutremur) / p_alarma

print("Probabilitatea ca a avut loc un cutremur stiind ca alarma de incendiu a fost declansata:", "%.4f" % p_cutremur_date_alarma)

# 3 
p_incendiu = 0.01
p_not_alarma_date_incendiu = 0.05
p_not_alarma = (0.98 * 0.0005) + (0.9999 * (1 - 0.0005))

p_incendiu_date_not_alarma = (p_incendiu * p_not_alarma_date_incendiu) / p_not_alarma

print("Probabilitatea ca un incendiu sa fi avut loc, fără ca alarma de incendiu să se activeze:", "%.4f" % p_incendiu_date_not_alarma)