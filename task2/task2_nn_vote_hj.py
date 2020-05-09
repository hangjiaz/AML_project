import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
import keras
from keras.callbacks import *
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from collections import Counter
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"



############## read data

x_traindata = pd.read_csv("X_train.csv", header=0)
y_traindata = pd.read_csv("y_train.csv", header=0)
testdata = pd.read_csv("X_test.csv", header=0)   	

y_testid = testdata.iloc[:,0].values

y_pred_mat = np.zeros((y_testid.shape[0], 9))




for k in range(9):
#############  downsampling in the majority class
    x_train = x_traindata.iloc[:,1:1001].values
    x_test = testdata.iloc[:,1:1001].values
    y_train = y_traindata.iloc[:,1].values
    np.random.seed(k)
    class1_idx = np.where(y_train == 1)[0]
    del_class1_idx = np.random.choice(class1_idx, size = len(class1_idx)-sum(y_train == 0), replace=False) 
    print('del_class1_idx:', del_class1_idx[:5])
    x_train = np.delete(x_train, (del_class1_idx), axis = 0) 
    y_train = np.delete(y_train, (del_class1_idx), axis = 0) 
    
     
    ############   data standardization 
    scaler = StandardScaler()
    x_train_norm = scaler.fit_transform(x_train)
    x_test_norm = scaler.transform(x_test)
    
    
    ############   remove outliers
    LOF = LocalOutlierFactor(n_neighbors=40, contamination=0.08)
    LOF.fit(x_train_norm)
    y_pred_local = LOF.fit_predict(x_train_norm)
    locations = np.where(y_pred_local == -1)
    
    rng = np.random.RandomState(42)
    IsoTree = IsolationForest(max_samples=100,random_state=rng,contamination=0.08)
    IsoTree.fit(x_train_norm)
    y_pred_iso = IsoTree.predict(x_train_norm)
    locations = np.where(y_pred_iso == -1)
    
    x_clean = x_train_norm
    y_clean = y_train
    
    for i in range(len(y_pred_local)-1,-1,-1):
        if ((y_pred_iso[i] == -1) and (y_pred_local[i] == -1)):
            x_clean = np.delete(x_clean, i, axis=0)
            y_clean = np.delete(y_clean, i, axis=0)
     
    
    ############## CV for paramter tuning
  
    
    ############# model construction
    model = Sequential()
    model.name = 'model'
    
    
    model.add(Dense(100, activation='relu', kernel_initializer='random_uniform', input_shape=(x_clean.shape[1],)))

        
    model.add(Dense(3, activation='softmax',kernel_initializer='random_uniform'))
    
    
    optim = keras.optimizers.Adadelta()
    
    
    model.compile(optimizer=optim,
          loss='categorical_crossentropy',
          metrics=['accuracy'])
    
################ model training     
    y_clean = keras.utils.to_categorical(y_clean, 3)
    model.fit(x_clean, y_clean, batch_size=300, epochs=100,  verbose=1)    
    y_pred_mat[:,k] = np.argmax(model.predict(x_test_norm), axis=1)


y_pred_vote = np.zeros(y_testid.shape[0])
for j in range(y_pred_mat.shape[0]):
    y_pred_vote[j] =  Counter(y_pred_mat[j]).most_common(1)[0][0]

print('vote_mat:',y_pred_mat[:5])
print('vote_result:', y_pred_vote[:5])



################### write output file
with open('output.csv', 'w') as f:
    f.write("{},{}\n".format("id", "y"))
    for i in range(len(y_testid)):
        f.write("{},{}\n".format(y_testid[i], y_pred_vote[i]))

