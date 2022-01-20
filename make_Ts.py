import random
import math
import numpy as np
from tqdm import tqdm
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import seaborn as sns

x = np.linspace(0,8,1000)
rannew = []
#aaa = [1,1,1,1,2,2,2,2,3,3,3,3,2,2,2,2,1,1,1,1]
aaa = [0,1,2,3,4,5,6,7,8,9,10,9,8,7,6,5,4,3,2,1,0]
for X in tqdm(range(21)):
    enen = random.random()*0
    ran = [random.random() for i in range(int(10000)*aaa[X])]
    ran2 = [(1+X/14)*(1-i**2)*np.pi for i in ran]
    ran2 = list(filter(lambda x: x >0.25*(1+X/15)*np.pi+enen, ran2))
    rannew = rannew + ran2

bw = 0.1
kde_model = gaussian_kde(rannew, bw_method=bw)
split = 1000
y = kde_model(x)

fig = plt.figure(figsize=(12,7))
ax1 = fig.add_subplot(111)
ax1.set_xlim(left=0, right=10.0)
ax1.hist(rannew, alpha=0.3, bins=100)
ax2 = ax1.twinx()
ax2.plot(x,y)
ax2.set_ylim(bottom=0, top=0.35)
plt.savefig('T(S).jpg')
plt.clf()



