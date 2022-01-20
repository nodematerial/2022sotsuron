import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

df = pd.read_csv('submission.csv')
df.expected_weight = df.expected_weight.apply(eval)
df.true = df.true.apply(eval)

for i in range(50):
    idx = i
    a, b = df.expected_weight.iloc[idx], df.true.iloc[idx]
    x = np.arange(49)
    plt.plot(x,b)
    plt.plot(x,a)
    plt.savefig('hikaku.jpg')
    time.sleep(1)
    plt.clf()