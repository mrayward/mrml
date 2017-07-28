import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from pyDOE import *
from MRML import *

df = pd.read_csv('./crime/crime2.csv')

val = df['ViolentCrimesPerPop '].values

cdf = Cdf(val)

fig = plt.figure()
ax = fig.add_subplot(111)

cdf.plot(ax=ax)

#px, sample = cdf.valuesample(500)

px, sample = cdf.valuesample(px=lhs(5, 1))
ax.plot(px, sample, 'o', color='red')

lhsample =lhs(5, 1)
print(lhsample)

plt.show()