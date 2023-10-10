import matplotlib.pyplot as plt
import numpy as np
import random
import arviz as az

np.random.seed(8231)
probabilitate_stema = 0.3
nr_experimente = 100
# nr_stema_obtinut = np.random.binomial(nr_experimente, probabilitate_stema) # de cate ori obtinem stema

def flipCoin(probabilitate_stema):
    if random.random() < probabilitate_stema:
        return "s"
    else:
        return "b"
    
def experiment(probabilitate_stema):
    flips_fair = [flipCoin(0.5) for _ in range(10)]
    flips_unfair = [flipCoin(probabilitate_stema) for i in range(10)]

    rezultat_experiment = "".join([flips_fair[i] + flips_unfair[i] for i in range(10)])
    # print(rezultat_experiment)
    return rezultat_experiment
    

# experiment(0.3)   
# print(nr_stema_obtinut)

rezultate = {"ss": 0, "sb": 0, "bs": 0, "bb": 0}

for i in range(nr_experimente):
    rezultat_experiment = experiment(probabilitate_stema)

    for j in range(0, len(rezultat_experiment), 2):
        outcome = rezultat_experiment[j : j + 2]
        rezultate[outcome] += 1
    

print(rezultate)
x = rezultate.keys()
y = rezultate.values()

plt.bar(x, y)
plt.show() 