import random
import math
import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

df = pd.DataFrame(index=[], columns=['weight','density'])

interval = 49
med = (interval+1)/2
sigma = 5+random.random()
density_interval = 100
iteration = 5000

def gauss(x):
    return np.exp(-(x-med)**2/(2*sigma**2))

def random_08_10(x):
    return (4+random.random())*gauss(x)/5

for i in tqdm(range(iteration)):
    #weight = [0,1,2,3,4,5,6,7,8,9,10,9,8,7,6,5,4,3,2,1,0]
    weight = [random_08_10(i) for i in range(interval)]
    weight = [i/sum(weight) for i in weight]
    num_random = [round(i*100000) for i in weight]

    r_min = 40
    r_max = 450
    all = []

    for idx, num in enumerate(num_random):
        r = r_min + (r_max - r_min)*idx/interval
        randoms = [random.random() for i in range(num)]
        adopted = [(2*i-i**2)*r for i in randoms]
        all = all + adopted

    all = np.array(sorted(list(filter(lambda x: x>r_min, all))))
    SUM = sum(all)

    x = np.linspace(r_min,r_max,density_interval)
    density = []
    for i in range(density_interval-1):
        density.append(np.count_nonzero((x[i]< all) & (all<x[i+1]))/SUM)

    df = df.append({'weight': weight, 'density': density}, ignore_index=True)

df.to_csv('artificial_data.csv', index=False)