import random
from pgmpy.models import *
from pgmpy.factors.discrete import TabularCPD
import networkx as nx


def aruncare_moneda(probabilitate):
    return random.random() < probabilitate # intoarcem numarul obtinut dupa aruncare

def game():
    P0_wins = 0 # nr de castiguri are lui P0
    P1_wins = 0 # nr de castiguri are lui P1

    for _ in range(20000):
        P0_incepe = aruncare_moneda(1/2) # probabilitatea ca P0 sa inceapa este 1/2
        P0_steme = aruncare_moneda(1/3) # probabilitatea ca P0 sa obtina stema este 1/3

        P0_steme_int = int(P0_steme) + 1 # numarul de aruncari pe care le executa P1
        P1_steme_aruncari = [] # lista cu output-urile aruncarilor lui P1

        # P1 arunca moneda de P0_steme + 1
        for _ in range(P0_steme_int):
            P1_steme_aruncari.append(aruncare_moneda(1/2))  # adaugam in lista rezultatul

        P1_steme = sum(P1_steme_aruncari) # nr total de steme obtinute de P1

        if P0_incepe:
            castigator_runda = P0_steme >= P1_steme
        else:
            castigator_runda = P0_steme > P1_steme

        if castigator_runda:
            P0_wins += 1
        else:
            P1_wins += 1

    # calculam procentajul de castig pentru fiecare
    procentaj_PO = (P0_wins / 20000) * 100
    procentaj_P1 = (P1_wins / 20000) * 100

    print(f"Jucatorul PO a castigat {procentaj_PO:.2f}% dintre jocuri.")
    print(f"Jucatorul P1 a castigat {procentaj_P1:.2f}% dintre jocuri.")

    # P0_incepe este parintele lui P0_steme
    # P1_steme este parintele lui P1_steme
    model = BayesianNetwork([('P0_incepe', 'P0_steme'), ('P0_steme', 'P1_steme')])

    # Definim modelul
    cpd_P0_incepe = TabularCPD(variable="P0_incepe", variable_card=2, values=[[1/2], [1/2]])
    cpd_P0_steme = TabularCPD(variable='P0_steme', variable_card=2, values=[[2/3, 1/3], [1/3, 2/3]],
                              evidence=['P0_incepe'], evidence_card=[2])
    cpd_P1_steme = TabularCPD(variable='P1_steme', variable_card=2, values=[[1/2, 1/2], [1/2, 1/2]],
                              evidence=['P0_steme'], evidence_card=[2])

    model.add_cpds(cpd_P0_incepe, cpd_P0_steme, cpd_P1_steme)
    print(cpd_P0_incepe)
    print(cpd_P0_steme)
    print(cpd_P1_steme)

if __name__ == "__main__":
    game()