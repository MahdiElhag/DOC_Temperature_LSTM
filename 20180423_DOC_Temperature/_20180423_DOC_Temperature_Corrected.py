#################################################################################################

#importing essiential libraries
import numpy as np
import pandas as pd
from pandas import DataFrame
from matplotlib import pyplot as plt
import time
from time import gmtime, strftime
import subprocess as sp
import os.path

import researchpy as rp

#################################################################################################

#importing keras (tensorflow backended)
from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout
from keras.layers import LSTM
from keras.callbacks import CSVLogger
from keras import optimizers
from keras.utils import plot_model

#################################################################################################

#cleaning DOS screen for new prompts
tmp = sp.call('cls',shell=True)

#################################################################################################

#importing sklern libraries
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

#################################################################################################

#training and validation parameters
data_percentage = 0.9  #0.90
iterations_numr = 4    #2
moving_avg_degr = 400  #1000
dropout_percent = 0.20  #0.15
lstm_time_stamp = 50    #50

#################################################################################################

#creating a record directory for the models and figues
naming = strftime("%Y_%m_%d_%H_%M_%S", gmtime())

current_dir = os.getcwd()
final_dir = os.path.join(current_dir, naming)
if not os.path.exists(final_dir):
    os.makedirs(final_dir)

#################################################################################################

#preparing a log to get the correlation results
log = open(naming + "/" + naming + ".txt", "w")

#################################################################################################

#defining a csv logger for training losses
csv_logger = CSVLogger(naming + "/" + naming + '.csv', append=True, separator=',')

#################################################################################################

#reading training and validation csv file from local
dft = pd.read_csv("DV_K08R2KS5ZA_Correlation.csv", index_col=False)

#################################################################################################

#results = rp.ttest(dft['sepal_width'], dft['sepal_width'])
results = rp.ttest(dft.DOC_US, dft.DOC_DS)
print(results)

#print(dft.corr())
#log.write(np.array_str(dft.corr()))
#log.write("\n\n")

plt.matshow(dft.corr())
plt.savefig(naming + '/' + 'Model_Loss.png')

log.close()

