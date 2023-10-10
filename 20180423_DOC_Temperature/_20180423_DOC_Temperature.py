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
log.write(naming + "\n\n")
log.write("data_percentage = " + str(data_percentage) + "\n")
log.write("moving_avg_degr = " + str(moving_avg_degr) + "\n")
log.write("dropout_percent = " + str(dropout_percent) + "\n")
log.write("lstm_time_stamp = " + str(lstm_time_stamp) + "\n")
log.write("iterations_numr = " + str(iterations_numr) + "\n\n")

#################################################################################################

#defining a csv logger for training losses
csv_logger = CSVLogger(naming + "/" + naming + '.csv', append=True, separator=',')

#################################################################################################

#reading training and validation csv file from local
dft = pd.read_csv("DV_K08R2KS5ZA_V2.csv", index_col=False)
dfv = pd.read_csv("20171101_0026_80000985_DO011580_MLOG.csv", index_col=False)


#################################################################################################

#sorting data in ascending form
dft.sort_index(ascending = True,inplace = True)

dft['DOC_DS_AVG'] = dft['DOC_DS'].rolling(window=moving_avg_degr).mean().fillna(0)
dft['DOC_US_AVG'] = dft['DOC_US'].rolling(window=moving_avg_degr).mean().fillna(0)
dft['EngSpd_AVG'] = dft['EngSpd'].rolling(window=moving_avg_degr).mean().fillna(0)
dft['Torque_AVG'] = dft['Torque'].rolling(window=moving_avg_degr).mean().fillna(0)
dft['ExhFlw_AVG'] = dft['ExhFlw'].rolling(window=moving_avg_degr).mean().fillna(0)
dft['IntPrs_AVG'] = dft['IntPrs'].rolling(window=moving_avg_degr).mean().fillna(0)
dft['APPsen_AVG'] = dft['APPsen'].rolling(window=moving_avg_degr).mean().fillna(0)
dft['CoTemp_AVG'] = dft['CoTemp'].rolling(window=moving_avg_degr).mean().fillna(0)
dft['ThrtDS_AVG'] = dft['ThrtDS'].rolling(window=moving_avg_degr).mean().fillna(0)
dft['CmprUS_AVG'] = dft['CmprUS'].rolling(window=moving_avg_degr).mean().fillna(0)
dft['Regenr_AVG'] = dft['Regenr'].rolling(window=moving_avg_degr).mean().fillna(0)
dft['EGRsen_AVG'] = dft['EGRsen'].rolling(window=moving_avg_degr).mean().fillna(0)

dfv.sort_index(ascending = True,inplace = True)

dfv['DOC_DS_AVG'] = dfv['DOC_DS'].rolling(window=moving_avg_degr).mean().fillna(0)
dfv['DOC_US_AVG'] = dfv['DOC_US'].rolling(window=moving_avg_degr).mean().fillna(0)
dfv['EngSpd_AVG'] = dfv['EngSpd'].rolling(window=moving_avg_degr).mean().fillna(0)
dfv['Torque_AVG'] = dfv['Torque'].rolling(window=moving_avg_degr).mean().fillna(0)
dfv['ExhFlw_AVG'] = dfv['ExhFlw'].rolling(window=moving_avg_degr).mean().fillna(0)
dfv['IntPrs_AVG'] = dfv['IntPrs'].rolling(window=moving_avg_degr).mean().fillna(0)
dfv['APPsen_AVG'] = dfv['APPsen'].rolling(window=moving_avg_degr).mean().fillna(0)
dfv['CoTemp_AVG'] = dfv['CoTemp'].rolling(window=moving_avg_degr).mean().fillna(0)
dfv['ThrtDS_AVG'] = dfv['ThrtDS'].rolling(window=moving_avg_degr).mean().fillna(0)
dfv['CmprUS_AVG'] = dfv['CmprUS'].rolling(window=moving_avg_degr).mean().fillna(0)
dfv['Regenr_AVG'] = dfv['Regenr'].rolling(window=moving_avg_degr).mean().fillna(0)
dfv['EGRsen_AVG'] = dfv['EGRsen'].rolling(window=moving_avg_degr).mean().fillna(0)

#################################################################################################

#normalizing csv data to be used in training and validation
scaler00 = MinMaxScaler(feature_range=(0, 1))
scaler01 = MinMaxScaler(feature_range=(0, 1))
scaler02 = MinMaxScaler(feature_range=(0, 1))
scaler03 = MinMaxScaler(feature_range=(0, 1))
scaler04 = MinMaxScaler(feature_range=(0, 1))
scaler05 = MinMaxScaler(feature_range=(0, 1))
scaler06 = MinMaxScaler(feature_range=(0, 1))
scaler07 = MinMaxScaler(feature_range=(0, 1))
scaler08 = MinMaxScaler(feature_range=(0, 1))
scaler09 = MinMaxScaler(feature_range=(0, 1))
scaler10 = MinMaxScaler(feature_range=(0, 1))
scaler11 = MinMaxScaler(feature_range=(0, 1))

dft['DOC_DS_Norm'] = scaler00.fit_transform(dft['DOC_DS_AVG'].reshape(-1,1))
dft['DOC_US_Norm'] = scaler01.fit_transform(dft['DOC_US_AVG'].reshape(-1,1))
dft['EngSpd_Norm'] = scaler02.fit_transform(dft['EngSpd_AVG'].reshape(-1,1))
dft['Torque_Norm'] = scaler03.fit_transform(dft['Torque_AVG'].reshape(-1,1))
dft['ExhFlw_Norm'] = scaler04.fit_transform(dft['ExhFlw_AVG'].reshape(-1,1))
dft['IntPrs_Norm'] = scaler05.fit_transform(dft['IntPrs_AVG'].reshape(-1,1))
dft['APPsen_Norm'] = scaler06.fit_transform(dft['APPsen_AVG'].reshape(-1,1))
dft['CoTemp_Norm'] = scaler07.fit_transform(dft['CoTemp_AVG'].reshape(-1,1))
dft['ThrtDS_Norm'] = scaler08.fit_transform(dft['ThrtDS_AVG'].reshape(-1,1))
dft['CmprUS_Norm'] = scaler09.fit_transform(dft['CmprUS_AVG'].reshape(-1,1))
dft['Regenr_Norm'] = scaler10.fit_transform(dft['Regenr_AVG'].reshape(-1,1))
dft['EGRsen_Norm'] = scaler11.fit_transform(dft['EGRsen_AVG'].reshape(-1,1))

dfv['DOC_DS_Norm'] = scaler00.fit_transform(dfv['DOC_DS_AVG'].reshape(-1,1))
dfv['DOC_US_Norm'] = scaler01.fit_transform(dfv['DOC_US_AVG'].reshape(-1,1))
dfv['EngSpd_Norm'] = scaler02.fit_transform(dfv['EngSpd_AVG'].reshape(-1,1))
dfv['Torque_Norm'] = scaler03.fit_transform(dfv['Torque_AVG'].reshape(-1,1))
dfv['ExhFlw_Norm'] = scaler04.fit_transform(dfv['ExhFlw_AVG'].reshape(-1,1))
dfv['IntPrs_Norm'] = scaler05.fit_transform(dfv['IntPrs_AVG'].reshape(-1,1))
dfv['APPsen_Norm'] = scaler06.fit_transform(dfv['APPsen_AVG'].reshape(-1,1))
dfv['CoTemp_Norm'] = scaler07.fit_transform(dfv['CoTemp_AVG'].reshape(-1,1))
dfv['ThrtDS_Norm'] = scaler08.fit_transform(dfv['ThrtDS_AVG'].reshape(-1,1))
dfv['CmprUS_Norm'] = scaler09.fit_transform(dfv['CmprUS_AVG'].reshape(-1,1))
dfv['Regenr_Norm'] = scaler10.fit_transform(dfv['Regenr_AVG'].reshape(-1,1))
dfv['EGRsen_Norm'] = scaler11.fit_transform(dfv['EGRsen_AVG'].reshape(-1,1))

#################################################################################################

#cleaning DOS screen for new prompts
tmp = sp.call('cls',shell=True)

#################################################################################################

#plottting the arrays (US & DS Temperatures)
plt.plot(dft['Time'], dft['DOC_US_AVG'], '-r',label = 'DOC_US_T')
plt.plot(dft['Time'], dft['DOC_DS_AVG'], '-b',label = 'DOC_DS_T')
plt.title("DOC Upstream & Downstream Tempertures in Training Data")
plt.xlabel("Time [s]")
plt.ylabel("Temperature [C]")
plt.legend(loc='best')
plt.savefig(naming + '/' + 'Fig_1.png')
#plt.show()
plt.clf()

#################################################################################################

#plottting the arrays Coolant Temperature
plt.plot(dft['Time'], dft['CoTemp_AVG'], '-r',label = 'CoTemp_AVG')
plt.title("Coolant Temperature in Training Data")
plt.xlabel("Time [s]")
plt.ylabel("Coolant Temperature [C]")
plt.legend(loc='best')
plt.savefig(naming + '/' + 'Fig_2.png')
#plt.show()
plt.clf()

#################################################################################################

#plottting the arrays of Throttle DS and Compressor US
plt.plot(dft['Time'], dft['ThrtDS_AVG'], '-r',label = 'ThrtDS_AVG')
plt.plot(dft['Time'], dft['CmprUS_AVG'], '-b',label = 'CmprUS_AVG')
plt.title("Throttle DS & Turbo US Temperature in Training Data")
plt.xlabel("Time [s]")
plt.ylabel("Temperature [C]")
plt.legend(loc='best')
plt.savefig(naming + '/' + 'Fig_3.png')
#plt.show()
plt.clf()

#################################################################################################

#plottting the arrays of APPsen and EGRsen
plt.plot(dft['Time'], dft['APPsen_AVG'], '-b',label = 'APPsen_AVG')
plt.plot(dft['Time'], dft['EGRsen_AVG'], '-r',label = 'EGRsen_AVG')
plt.title("APP sensor & EGR sensor Measurements in Training Data")
plt.xlabel("Time [s]")
plt.ylabel("Perecentage [%]")
plt.legend(loc='best')
plt.savefig(naming + '/' + 'Fig_4.png')
#plt.show()
plt.clf()

#################################################################################################

#plottting the arrays of exhaust flowrate
plt.title("Exhasut Flowrate Measurements in Training Data")
plt.plot(dft['Time'], dft['ExhFlw_AVG'], '-b',label = 'ExhFlw_AVG')
plt.xlabel("Time [s]")
plt.ylabel("Flowrate [m3/s]")
plt.legend(loc='best')
plt.savefig(naming + '/' + 'Fig_5.png')
#plt.show()
plt.clf()

#################################################################################################

#plottting the arrays of manifold intake pressure
plt.title("Manifold Pressure Measurements in Training Data")
plt.plot(dft['Time'], dft['IntPrs_AVG'], '-b',label = 'IntPrs_AVG')
plt.xlabel("Time [s]")
plt.ylabel("Pressure [hPa]")
plt.legend(loc='best')
plt.savefig(naming + '/' + 'Fig_6.png')
#plt.show()
plt.clf()

#################################################################################################

#plottting the arrays of engine speed
plt.title("Engine Speed Measurements in Training Data")
plt.plot(dft['Time'], dft['EngSpd_AVG'], '-b',label = 'EngSpd_AVG')
plt.xlabel("Time [s]")
plt.ylabel("Engine Speed [rpm]")
plt.legend(loc='best')
plt.savefig(naming + '/' + 'Fig_7.png')
#plt.show()
plt.clf()

#################################################################################################

#plottting the arrays of Torque
plt.title("Torque Measurements in Training Data")
plt.plot(dft['Time'], dft['Torque_AVG'], '-b',label = 'Torque_AVG')
plt.xlabel("Time [s]")
plt.ylabel("Torque [Nm]")
plt.legend(loc='best')
plt.savefig(naming + '/' + 'Fig_8.png')
#plt.show()
plt.clf()

#################################################################################################

#plottting the arrays of regenration periods
plt.title("Regeneration Measurement in Training Data")
plt.plot(dft['Time'], dft['Regenr_AVG'], '-b',label = 'Regenr_AVG')
plt.xlabel("Time [s]")
plt.legend(loc='best')
plt.savefig(naming + '/' + 'Fig_9.png')
#plt.show()
plt.clf()

#################################################################################################

#checking Pearsons Correlation between features and DOC_DS 
log.write("Correlation between DOC_US and DOC_DS" + "\n")
log.write(np.array_str(np.corrcoef(dft.DOC_US,dft.DOC_DS)))
log.write("\n\n")

log.write("Correlation between EngSpd and DOC_DS" + "\n")
log.write(np.array_str(np.corrcoef(dft.EngSpd,dft.DOC_DS)))
log.write("\n\n")

log.write("Correlation between Torque and DOC_DS" + "\n")
log.write(np.array_str(np.corrcoef(dft.Torque,dft.DOC_DS)))
log.write("\n\n")

log.write("Correlation between ExhFlw and DOC_DS" + "\n")
log.write(np.array_str(np.corrcoef(dft.ExhFlw,dft.DOC_DS)))
log.write("\n\n")

log.write("Correlation between IntPrs and DOC_DS" + "\n")
log.write(np.array_str(np.corrcoef(dft.IntPrs,dft.DOC_DS)))
log.write("\n\n")

log.write("Correlation between APPsen and DOC_DS" + "\n")
log.write(np.array_str(np.corrcoef(dft.APPsen,dft.DOC_DS)))
log.write("\n\n")

log.write("Correlation between CoTemp and DOC_DS" + "\n")
log.write(np.array_str(np.corrcoef(dft.CoTemp,dft.DOC_DS)))
log.write("\n\n")

log.write("Correlation between ThrtDS and DOC_DS" + "\n")
log.write(np.array_str(np.corrcoef(dft.ThrtDS,dft.DOC_DS)))
log.write("\n\n")

log.write("Correlation between CmprUS and DOC_DS" + "\n")
log.write(np.array_str(np.corrcoef(dft.CmprUS,dft.DOC_DS)))
log.write("\n\n")

log.write("Correlation between Regenr and DOC_DS" + "\n")
log.write(np.array_str(np.corrcoef(dft.Regenr,dft.DOC_DS)))
log.write("\n\n")

#checking Pearsons Correlation between features and DOC_US 
log.write("Correlation between DOC_DS and DOC_US" + "\n")
log.write(np.array_str(np.corrcoef(dft.DOC_DS,dft.DOC_US)))
log.write("\n\n")

log.write("Correlation between EngSpd and DOC_US" + "\n")
log.write(np.array_str(np.corrcoef(dft.EngSpd,dft.DOC_US)))
log.write("\n\n")

log.write("Correlation between Torque and DOC_US" + "\n")
log.write(np.array_str(np.corrcoef(dft.Torque,dft.DOC_US)))
log.write("\n\n")

log.write("Correlation between ExhFlw and DOC_US" + "\n")
log.write(np.array_str(np.corrcoef(dft.ExhFlw,dft.DOC_US)))
log.write("\n\n")

log.write("Correlation between IntPrs and DOC_US" + "\n")
log.write(np.array_str(np.corrcoef(dft.IntPrs,dft.DOC_US)))
log.write("\n\n")

log.write("Correlation between APPsen and DOC_US" + "\n")
log.write(np.array_str(np.corrcoef(dft.APPsen,dft.DOC_US)))
log.write("\n\n")

log.write("Correlation between CoTemp and DOC_US" + "\n")
log.write(np.array_str(np.corrcoef(dft.CoTemp,dft.DOC_US)))
log.write("\n\n")

log.write("Correlation between ThrtDS and DOC_US" + "\n")
log.write(np.array_str(np.corrcoef(dft.ThrtDS,dft.DOC_US)))
log.write("\n\n")

log.write("Correlation between CmprUS and DOC_US" + "\n")
log.write(np.array_str(np.corrcoef(dft.CmprUS,dft.DOC_US)))
log.write("\n\n")

log.write("Correlation between Regenr and DOC_US" + "\n")
log.write(np.array_str(np.corrcoef(dft.Regenr,dft.DOC_US)))
log.write("\n\n")

#################################################################################################

#taking 90% of the data points as train. This number can change.
train_size = int(data_percentage*len(dft))

#segregating the inputs and ouput on the test and train data
trainX = dft.loc[1:train_size,['EngSpd_Norm', 'Torque_Norm', 'CoTemp_Norm','ExhFlw_Norm', 'IntPrs_Norm', 'APPsen_Norm', 'ThrtDS_Norm','CmprUS_Norm', 'Regenr_Norm', 'EGRsen_Norm']]
trainY = dft.loc[1:train_size,['DOC_DS_Norm', 'DOC_US_Norm']]
log.write("trainX = dft.loc[1:train_size,['EngSpd_Norm', 'Torque_Norm', 'CoTemp_Norm','ExhFlw_Norm', 'IntPrs_Norm', 'APPsen_Norm', 'ThrtDS_Norm','CmprUS_Norm', 'Regenr_Norm', 'EGRsen_Norm']]" + "\n")
log.write("trainY = dft.loc[1:train_size,['DOC_DS_Norm', 'DOC_US_Norm']]" + "\n\n")

testX1 = dft.loc[1:train_size,['EngSpd_Norm', 'Torque_Norm', 'CoTemp_Norm','ExhFlw_Norm', 'IntPrs_Norm', 'APPsen_Norm', 'ThrtDS_Norm','CmprUS_Norm', 'Regenr_Norm', 'EGRsen_Norm']]
testY1 = dft.loc[1:train_size,['DOC_DS_Norm', 'DOC_US_Norm']]
log.write("testX1 = dft.loc[1:train_size,['EngSpd_Norm', 'Torque_Norm', 'CoTemp_Norm','ExhFlw_Norm', 'IntPrs_Norm', 'APPsen_Norm', 'ThrtDS_Norm','CmprUS_Norm', 'Regenr_Norm', 'EGRsen_Norm']]" + "\n")
log.write("testY1 = dft.loc[1:train_size,['DOC_DS_Norm', 'DOC_US_Norm']]" + "\n")

#testX2 = dft.loc[train_size:len(dft),['EngSpd_Norm', 'Torque_Norm', 'CoTemp_Norm','ExhFlw_Norm', 'IntPrs_Norm', 'APPsen_Norm', 'ThrtDS_Norm','CmprUS_Norm', 'Regenr_Norm', 'EGRsen_Norm']]
#testY2 = dft.loc[train_size:len(dft),['DOC_DS_Norm', 'DOC_US_Norm']]
#log.write("testX2 = dft.loc[train_size:len(dft),['EngSpd_Norm', 'Torque_Norm', 'CoTemp_Norm','ExhFlw_Norm', 'IntPrs_Norm', 'APPsen_Norm', 'ThrtDS_Norm','CmprUS_Norm', 'Regenr_Norm', 'EGRsen_Norm']]" + "\n")
#log.write("testY2 = dft.loc[train_size:len(dft),['DOC_DS_Norm', 'DOC_US_Norm']]" + "\n\n")

testX2 = dfv.loc[1:len(dfv),['EngSpd_Norm', 'Torque_Norm', 'CoTemp_Norm','ExhFlw_Norm', 'IntPrs_Norm', 'APPsen_Norm', 'ThrtDS_Norm','CmprUS_Norm', 'Regenr_Norm', 'EGRsen_Norm']]
testY2 = dft.loc[1:len(dfv),['DOC_DS_Norm', 'DOC_US_Norm']]
log.write("testX2 = dft.loc[1:len(dfv),['EngSpd_Norm', 'Torque_Norm', 'CoTemp_Norm','ExhFlw_Norm', 'IntPrs_Norm', 'APPsen_Norm', 'ThrtDS_Norm','CmprUS_Norm', 'Regenr_Norm', 'EGRsen_Norm']]" + "\n")
log.write("testY2 = dft.loc[1:len(dfv),['DOC_DS_Norm', 'DOC_US_Norm']]" + "\n\n")

#################################################################################################

#The inputs needed to be reshaped in the format of  a 3d Tensor with dimesnions = [batchsize,timesteps,features]
trainX = np.reshape(np.array(trainX),(trainX.shape[0],1,trainX.shape[1]))
testX1 = np.reshape(np.array(testX1),(testX1.shape[0],1,testX1.shape[1]))
testX2 = np.reshape(np.array(testX2),(testX2.shape[0],1,testX2.shape[1]))

#################################################################################################

#structuring the LSTM RNN network
model = Sequential()
model.add(LSTM(lstm_time_stamp ,batch_input_shape=(1,trainX.shape[1],trainX.shape[2]),return_sequences = True))
model.add(Dropout(dropout_percent))
model.add(LSTM(lstm_time_stamp))
model.add(Dense(2))
#adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
adam = optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='mean_squared_error', optimizer=adam)
epochs = iterations_numr
start = time.time()
m = model.fit(trainX, np.array(trainY), epochs = epochs, batch_size=1, verbose=2,validation_split=0.1, callbacks=[csv_logger])
print ("Compilation Time : ", time.time() - start)
log.write("Training Time : " + str(time.time() - start))

#closing the log file
log.close()

#################################################################################################

#save model in JSON format
model.save(naming + "/" + naming + ".h5")
print("Model is saved model to disk")

#################################################################################################

# summarizing model for loss
plt.plot(m.history['loss'])
plt.plot(m.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss-Mean Squared Error')
plt.xlabel('epoch')
plt.legend(['Train', 'Valid'], loc='best')
plt.savefig(naming + '/' + 'Model_Loss.png')
#plt.show()
plt.clf()

#################################################################################################

#comparing actual and predicted DOC_DS in training data by plotting
testPredict = model.predict(testX1,batch_size = 1)
plt.plot(scaler00.inverse_transform(testPredict[:,0].reshape(-1,1)),'-r',label = 'Model DOC_DS_T')
plt.plot(scaler00.inverse_transform(testY1.DOC_DS_Norm.reshape(-1,1)),'-b',label = 'Actual DOC_DS_T')    # Reshaped

#cleaning DOS screen for new prompts
tmp = sp.call('cls',shell=True)

plt.title("Actual & Predicted DOC Downstream Temperature (Training)")
plt.xlabel("Time [s]")
plt.ylabel("Temperature [C]")
plt.legend(loc='best')
plt.savefig(naming + '/' + 'DS_Train_Output.png')
#plt.show()
plt.clf()

#################################################################################################

#comparing actual and predicted DOC_US in training data by plotting
testPredict = model.predict(testX1,batch_size = 1)
plt.plot(scaler00.inverse_transform(testPredict[:,1].reshape(-1,1)),'-r',label = 'Model DOC_US_T')
plt.plot(scaler00.inverse_transform(testY1.DOC_US_Norm.reshape(-1,1)),'-b',label = 'Actual DOC_US_T')    # Reshaped

#cleaning DOS screen for new prompts
tmp = sp.call('cls',shell=True)

plt.title("Actual & Predicted DOC Upstream Temperature (Training)")
plt.xlabel("Time [s]")
plt.ylabel("Temperature [C]")
plt.legend(loc='best')
plt.savefig(naming + '/' + 'US_Train_Output.png')
#plt.show()
plt.clf()

#################################################################################################
#comparing actual and predicted DOC_DS in validation data by plotting
testPredict = model.predict(testX2,batch_size = 1)
plt.plot(scaler00.inverse_transform(testPredict[:,0].reshape(-1,1)),'-r',label = 'Model DOC_US_T')
plt.plot(scaler00.inverse_transform(testY2.DOC_DS_Norm.reshape(-1,1)),'-b',label = 'Actual DOC_US_T')    # Reshaped

#cleaning DOS screen for new prompts
tmp = sp.call('cls',shell=True)

plt.title("Actual & Predicted DOC Downstream Temperature (Validation)")
plt.xlabel("Time [s]")
plt.ylabel("Temperature [C]")
plt.legend(loc='best')
plt.savefig(naming + '/' + 'DS_Valid_Output.png')
#plt.show()
plt.clf()

#################################################################################################
#comparing actual and predicted DOC_US in validation data by plotting
testPredict = model.predict(testX2,batch_size = 1)
plt.plot(scaler00.inverse_transform(testPredict[:,1].reshape(-1,1)),'-r',label = 'Model DOC_US_T')
plt.plot(scaler00.inverse_transform(testY2.DOC_US_Norm.reshape(-1,1)),'-b',label = 'Actual DOC_US_T')    # Reshaped

#cleaning DOS screen for new prompts
tmp = sp.call('cls',shell=True)

plt.title("Actual & Predicted DOC Upstream Temperature (Validation)")
plt.xlabel("Time [s]")
plt.ylabel("Temperature [C]")
plt.legend(loc='best')
plt.savefig(naming + '/' + 'Valid_Output.png')
#plt.show()
plt.clf()
#################################################################################################
#transforming back the original predictions
#scaler00.inverse_transform(testPredict)