import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('results.csv', names=['threads', 'time'])
df = df.sort_values('threads')
t1 = df.loc[df['threads'] == 1, 'time'].values[0]

df['speedup'] = t1 / df['time']
df['ideal'] = df['threads']

plt.plot(df['threads'], df['speedup'], 'o-', label='Measured speedup')
plt.plot(df['threads'], df['ideal'], 'r--', label='Ideal linear speedup')
plt.xlabel('Number of threads')
plt.ylabel('Speedup')
plt.legend()
plt.grid(True)
plt.show()
