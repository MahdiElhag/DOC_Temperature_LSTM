import scipy.signal as signal
import pandas as pd
from pandas import DataFrame
from matplotlib import pyplot as plt
import numpy as np

df = pd.read_csv("DV_K08R2KS5ZA_Training.csv", index_col=False)
#DV_K08R2KS5ZA_Training
#20171101_0026_80000985_DO011580_MLOG
df.sort_index(ascending = True,inplace = True)

moving_avg_degr = 200

df['DOC_DS_AVG'] = df['DOC_DS'].rolling(window=moving_avg_degr).mean().fillna(0)
df['DOC_DS_DIF'] = df['DOC_DS'] - df['DOC_DS_AVG']

df['DOC_DS_SFT'] = df['DOC_DS'].shift(0)

plt.plot(df.loc[moving_avg_degr:90000,['DOC_DS_SFT']],'r-',label = 'Unfiltered')
plt.plot(df.loc[moving_avg_degr:90000,['DOC_DS_AVG']],'b-',label = 'Filtered')

plt.title("Filtered and Unfiltered DOC Downstream Temperature")
plt.xlabel("Time [ms]")
plt.ylabel("DOC Downstream Temperature [C]")
plt.legend(loc='best')
#plt.grid(which='both')

plt.show()
plt.clf()

plt.plot(df['DOC_DS_DIF'],'g-')
plt.show()
df.to_csv('20180627_FILTER_EFFECT.csv')

print(np.corrcoef(df.DOC_DS[60000:80000], df.DOC_DS_AVG[60000:80000]))
