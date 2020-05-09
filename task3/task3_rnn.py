import numpy as np
import pandas as pd
from biosppy.signals import ecg 
import math
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
import keras
from keras import backend as K
from keras.models import Model, Sequential
from keras.layers import *
from keras.callbacks import *
from collections import Counter
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from lightgbm import LGBMClassifier
import os

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

################ read data
y_traindata = pd.read_csv("y_train.csv", header=0)   
y_train = y_traindata.iloc[:,1].values
testdata = pd.read_csv("X_test.csv", header=0)  
y_testid = testdata.iloc[:,0].values

x_train = []
with open("X_train.csv") as f_train:
    for line in f_train.readlines()[1:]:
        s = list(map(int, line.split(',')[1:]))
        if len(s) < 17813:
            s.extend([0 for x in range(len(s), 17813)])
        x_train.append(s)
x_train = np.array(x_train)

x_test = []
with open("X_test.csv") as f_train:
    for line in f_train.readlines()[1:]:
        s = list(map(int, line.split(',')[1:]))
        if len(s) < 17813:
            s.extend([0 for x in range(len(s), 17813)])
        x_test.append(s)
x_test = np.array(x_test)

################# split data
x_ktrain, x_ktest, y_ktrain, y_ktest = train_test_split(x_train, y_train, test_size=0.4, random_state=0)



##################  feature extraction
def feature_extraction(x):
    X = []

    [ts, filtered_sig, rpeaks, temp_ts, temp, hr_ts, heart_rate]  = ecg.ecg(signal = x, sampling_rate=300, show=False)
    rpeaks = ecg.correct_rpeaks(signal=x, rpeaks=rpeaks, sampling_rate=300, tol=0.1)
    
    peaks = x[rpeaks]
    if len(heart_rate) < 2:
        heart_rate = [0, 1]
    if len(hr_ts) < 2:
        hr_ts = [0, 1]
    
    X.append(np.median(peaks))
    X.append(np.min(peaks))
    X.append(np.max(peaks))
    X.append(np.median(np.diff(rpeaks)))
    X.append(np.min(np.diff(rpeaks)))
    X.append(np.max(np.diff(rpeaks)))
    X.append(np.std(np.diff(rpeaks)))
    X.append(np.median(heart_rate))
    X.append(np.min(heart_rate))
    X.append(np.max(heart_rate))
    X.append(np.median(np.diff(heart_rate)))
    X.append(np.min(np.diff(heart_rate)))
    X.append(np.max(np.diff(heart_rate)))
    X.append(np.std(np.diff(heart_rate)))
    X.append(np.min(np.diff(hr_ts)))
    X.append(np.max(np.diff(hr_ts)))
    X.append(np.std(np.diff(hr_ts)))
    
#    X += list(np.median(temp, axis=0))
#    X += list(np.min(temp, axis=0))
#    X += list(np.max(temp, axis=0))

    

    return np.array(X)

train_data = np.apply_along_axis(feature_extraction, 1, x_ktrain)
test_data = np.apply_along_axis(feature_extraction, 1, x_ktest)

def temp_extraction(x):
   [ts, filtered_sig, rpeaks, temp_ts, temp, hr_ts, heart_rate]  = ecg.ecg(signal = x, sampling_rate=300, show=False)
   return  np.median(temp, axis=0)

train_temp = np.apply_along_axis(temp_extraction, 1, x_ktrain)
test_temp = np.apply_along_axis(temp_extraction, 1, x_ktest)


scaler = StandardScaler()
x_train_temp_stand = scaler.fit_transform(train_temp).reshape((train_temp.shape[0],180,1))
x_test_temp_stand = scaler.transform(test_temp).reshape((test_temp.shape[0],180,1))



    
y_ktrain_trans = keras.utils.to_categorical(y_ktrain, 4)
nn_input = Input((180,1,))
nn_1 = Sequential()
nn_1.add(LSTM(1, activation = 'relu',return_sequences= True, return_state= False))
nn_1.add(Flatten())

temp_dense = nn_1(nn_input)
nn_2 = Sequential()
nn_2.add(Dense(4, activation='softmax'))

nn_output = nn_2(temp_dense)

nn = Model(nn_input, nn_output )
optim = keras.optimizers.Adadelta()



nn.compile(optimizer=optim,
          loss='categorical_crossentropy',
          metrics=['accuracy'])


nn.fit(x_train_temp_stand, y_ktrain_trans, epochs=80, verbose=1)

train_temp_extracted = nn_1.predict(x_train_temp_stand)
test_temp_extracted = nn_1.predict(x_test_temp_stand)
print(train_temp_extracted[0])


################################

train_data = np.concatenate((train_data, train_temp_extracted), axis = 1)
test_data = np.concatenate((test_data,test_temp_extracted), axis = 1  )

x_train_stand = scaler.fit_transform(train_data)
x_test_stand = scaler.transform(test_data)
#
#
#
#
lr = LogisticRegression(random_state=0,penalty='l1').fit(x_train_stand, y_ktrain)
model = SelectFromModel(lr, prefit=True)
x_train_selected = model.transform(x_train_stand)
x_test_selected   = model.transform(x_test_stand)
#
#
#    
#    
clf = LGBMClassifier(boosting_type ='gbdt', random_state=0, n_estimators=100, num_leaves=50, max_depth = 20, reg_lambda = 0.01).fit(x_train_selected,y_ktrain)
#clf = SVC(C = 3, kernel = 'rbf',decision_function_shape ='ovr', gamma = 'auto')
#clf.fit(x_train_stand,y_ktrain)
y_kpred  = clf.predict(x_test_selected)
score = f1_score(y_ktest, y_kpred, average='micro')
print(score)   
#
#
#with open('output.csv', 'w') as f:
#    f.write("{},{}\n".format("id", "y"))
#    for i in range(len(y_testid)):
#        f.write("{},{}\n".format(y_testid[i], y_pred[i]))
#
#
#
