#################################################################################################

#importing essiential libraries
import numpy as np
import pandas as pd
import subprocess as sp
from matplotlib import pyplot as plt

#################################################################################################

#importing keras (tensorflow backended) and sklearn
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model

#################################################################################################

#defining validation parameters
validation_strt = 200#415983
validation_fnsh = 415983#573310 #validation_strt + 300 #319800
moving_avg_degr = 700 #300

#################################################################################################

#reading validation csv file from local
df = pd.read_csv("DV_K08R2KS5ZA_Training.csv", index_col=False)

#################################################################################################

#sorting data in ascending form
df.sort_index(ascending = True,inplace = True)

df['DOC_DS_AVG'] = df['DOC_DS'].rolling(window=moving_avg_degr).mean().fillna(0)
df['DOC_US_AVG'] = df['DOC_US'].rolling(window=moving_avg_degr).mean().fillna(0)
df['EngSpd_AVG'] = df['EngSpd'].rolling(window=moving_avg_degr).mean().fillna(0)
df['Torque_AVG'] = df['Torque'].rolling(window=moving_avg_degr).mean().fillna(0)
df['ExhFlw_AVG'] = df['ExhFlw'].rolling(window=moving_avg_degr).mean().fillna(0)
df['IntPrs_AVG'] = df['IntPrs'].rolling(window=moving_avg_degr).mean().fillna(0)
df['APPsen_AVG'] = df['APPsen'].rolling(window=moving_avg_degr).mean().fillna(0)
df['CoTemp_AVG'] = df['CoTemp'].rolling(window=moving_avg_degr).mean().fillna(0)
df['ThrtDS_AVG'] = df['ThrtDS'].rolling(window=moving_avg_degr).mean().fillna(0)
df['CmprUS_AVG'] = df['CmprUS'].rolling(window=moving_avg_degr).mean().fillna(0)
df['Regenr_AVG'] = df['Regenr'].rolling(window=moving_avg_degr).mean().fillna(0)
df['EGRsen_AVG'] = df['EGRsen'].rolling(window=moving_avg_degr).mean().fillna(0)

#################################################################################################

#normalizing csv data to be used in validation
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

df['DOC_DS_Norm'] = scaler00.fit_transform(df['DOC_DS_AVG'].reshape(-1,1))
df['DOC_US_Norm'] = scaler01.fit_transform(df['DOC_US_AVG'].reshape(-1,1))
df['EngSpd_Norm'] = scaler02.fit_transform(df['EngSpd_AVG'].reshape(-1,1))
df['Torque_Norm'] = scaler03.fit_transform(df['Torque_AVG'].reshape(-1,1))
df['ExhFlw_Norm'] = scaler04.fit_transform(df['ExhFlw_AVG'].reshape(-1,1))
df['IntPrs_Norm'] = scaler05.fit_transform(df['IntPrs_AVG'].reshape(-1,1))
df['APPsen_Norm'] = scaler06.fit_transform(df['APPsen_AVG'].reshape(-1,1))
df['CoTemp_Norm'] = scaler07.fit_transform(df['CoTemp_AVG'].reshape(-1,1))
df['ThrtDS_Norm'] = scaler08.fit_transform(df['ThrtDS_AVG'].reshape(-1,1))
df['CmprUS_Norm'] = scaler09.fit_transform(df['CmprUS_AVG'].reshape(-1,1))
df['Regenr_Norm'] = scaler10.fit_transform(df['Regenr_AVG'].reshape(-1,1))
df['EGRsen_Norm'] = scaler11.fit_transform(df['EGRsen_AVG'].reshape(-1,1))

#################################################################################################

#cleaning DOS screen for new prompts
tmp = sp.call('cls',shell=True)

#################################################################################################

#loading the saved model trained and ready
loading = input("Input the title of the model to be loaded: ")
model = load_model(loading + "/" + loading + ".h5")

#################################################################################################

#determining the prt of the csv data to be used and reshaping it
#testX = df.loc[validation_strt:validation_fnsh,['EngSpd_Norm','CoTemp_Norm','ExhFlw_Norm','ThrtDS_Norm','CmprUS_Norm', 'DOC_US_Norm']]
testX = df.loc[validation_strt:validation_fnsh,['EngSpd_Norm', 'Torque_Norm', 'CoTemp_Norm','ExhFlw_Norm', 'IntPrs_Norm', 'APPsen_Norm', 'ThrtDS_Norm','CmprUS_Norm', 'Regenr_Norm', 'EGRsen_Norm']]
testY = df.loc[validation_strt:validation_fnsh,['DOC_DS_Norm', 'DOC_US_Norm']]
testX= np.reshape(np.array(testX),(testX.shape[0],1,testX.shape[1]))

#################################################################################################

#cleaning DOS screen for new prompts
tmp = sp.call('cls',shell=True)

#################################################################################################

#comparing actual and predicted measurements in validation data by plotting
testPredict = model.predict(testX,batch_size = 1)
#plt.plot(scaler00.inverse_transform(testPredict),'-r',label = 'Model DOC_DS_T')
plt.plot(scaler00.inverse_transform(testPredict[:,1].reshape(-1,1)),'-r',label = 'Model DOC_DS_T')
#plt.plot(scaler00.inverse_transform(testPredict),'-r',label = 'Model DOC_DS_T')
plt.plot(scaler00.inverse_transform(testY.DOC_DS_Norm.reshape(-1,1)),'-b',label = 'Actual DOC_DS_T')    # Reshaped


prediction = pd.DataFrame(testPredict[:,1], columns=['DOC_DS_PRED_NORM']).to_csv('DOC_DS_PRD.csv')
prediction = pd.DataFrame(testY.DOC_DS_Norm.reshape(-1,1), columns=['DOC_DS_NORM']).to_csv('DOC_DS_ORG.csv')

prediction = pd.DataFrame(testPredict[:,0], columns=['DOC_US_PRED_NORM']).to_csv('DOC_US_PRD.csv')
prediction = pd.DataFrame(testY.DOC_US_Norm.reshape(-1,1), columns=['DOC_US_NORM']).to_csv('DOC_US_ORG.csv')

#cleaning DOS screen for new prompts
tmp = sp.call('cls',shell=True)

plt.title("Actual & Predicted DOC Upstream Temperature (Validation)")
plt.xlabel("Time [s]")
plt.ylabel("Temperature [C]")
plt.legend(loc='best')
plt.show()
plt.clf()

print(np.array_str(np.corrcoef(testY.DOC_DS_Norm,testPredict[:,1])))
print(np.array_str(np.corrcoef(testY.DOC_US_Norm,testPredict[:,0])))

#################################################################################################

#transforming back the original predictions
#scaler0.inverse_transform(testPredict)
