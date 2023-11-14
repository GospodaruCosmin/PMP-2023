import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('auto-mpg.csv')
df = df[df['horsepower'] != '?']  # eliminam randurile ce contin '?'

df['horsepower'] = pd.to_numeric(df['horsepower'])

plt.figure(figsize=(10, 6))
plt.scatter(df['horsepower'], df['mpg'], alpha=0.7)
plt.title('Relatia de dependenta dintre CP si mpg')
plt.xlabel('Cai putere (CP)')
plt.ylabel('Consum (mpg)')
plt.grid(True)
plt.show()