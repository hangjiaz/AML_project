#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from biosppy.signals import *
import matplotlib.pyplot as plt
import keras
from keras import backend as K
from keras.models import Model, Sequential
from sklearn.preprocessing import StandardScaler
from keras.layers import *
from keras.callbacks import *
from sklearn.metrics import balanced_accuracy_score
from sklearn.svm import SVC
from collections import Counter
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


# In[2]:


def standardization(x):
    x -= np.mean(x)  
    x /= np.std(x)
    return x
    
    


# In[3]:


y_traindata = pd.read_csv("train_labels.csv", header=0)   
y_train = y_traindata.iloc[:,1].values

testdata = pd.read_csv("test_eeg1.csv", header=0)   
y_testid = testdata.iloc[:,0].values


# In[4]:



def read_data(subject, file):
    x_eeg1 = []
    with open("{}_eeg1.csv".format(file)) as f:
        for line in f.readlines()[21600*(subject-1)+1:21600*subject+1]:
            s = list(map(float, line.split(',')[1:]))
            x_eeg1.append(s)
    x_eeg1= np.array(x_eeg1)

    x_eeg2 = []
    with open("{}_eeg2.csv".format(file)) as f:
        for line in f.readlines()[21600*(subject-1)+1:21600*subject+1]:
            s = list(map(float, line.split(',')[1:]))
            x_eeg2.append(s)
    x_eeg2  = np.array(x_eeg2)

    x_emg = []
    with open("{}_emg.csv".format(file)) as f:
        for line in f.readlines()[21600*(subject-1)+1:21600*subject+1]:
            s = list(map(float, line.split(',')[1:]))
            x_emg.append(s)
    x_emg= np.array(x_emg)
    return  x_eeg1,x_eeg2,x_emg 


# In[5]:


x_train_1_eeg1,  x_train_1_eeg2, x_train_1_emg  = read_data(1,'train')
x_train_2_eeg1,  x_train_2_eeg2, x_train_2_emg  = read_data(2,'train')
x_train_3_eeg1,  x_train_3_eeg2, x_train_3_emg  = read_data(3, 'train')


# In[ ]:


# x_test_1_eeg1,  x_test_1_eeg2, x_test_1_emg  = read_data(1, 'test')
# x_test_2_eeg1,  x_test_2_eeg2, x_test_2_emg  = read_data(2, 'test')


# In[76]:


label_1 = y_train[0:21600]-1
label_2 = y_train[21600:43200]-1
label_3 = y_train[43200:64800]-1


# In[83]:



def eeg_feature_extraction(x):
    X = np.zeros((512))
    [ts, filtered_sig, features_ts,theta,alpha_low, alpha_high,beta, gamma, plf_pairs, plf]  = eeg.eeg(signal=x, sampling_rate=128.0, show=False)    
    X = filtered_sig.reshape(512)

        
    return X



def apply_fun(x):
    result = []
    for i in range(x.shape[0]):
        result.append(eeg_feature_extraction(x[i]))
    result = np.array(result)
    return result

def concatenate_eeg(x1, x2):
    eeg = np.zeros((x1.shape[0], x1.shape[1], 2))
    for i in range(x1.shape[0]):
        eeg[i,:,0] = x1[i]
        eeg[i,:,1] = x2[i]
    return eeg

def concatenate_total(eeg, emg):
    data = np.zeros((eeg.shape[0], eeg.shape[1], 3))
    for i in range(eeg.shape[0]):
        data[i,:,:2] = eeg[i]
        data[i,:,2] = emg[i]
    return data


# In[38]:



x_train_1_eeg1_extracted = apply_fun(x_train_1_eeg1.reshape(21600, 512, 1))
x_train_1_eeg2_extracted = apply_fun(x_train_1_eeg2.reshape(21600, 512, 1))
x_train_1_eeg = concatenate_eeg(x_train_1_eeg1_extracted , x_train_1_eeg2_extracted)
x_train_1_eeg_stand = standardization(x_train_1_eeg)


x_train_2_eeg1_extracted = apply_fun(x_train_2_eeg1.reshape(21600, 512, 1))
x_train_2_eeg2_extracted = apply_fun(x_train_2_eeg2.reshape(21600, 512, 1))
x_train_2_eeg = concatenate_eeg(x_train_2_eeg1_extracted , x_train_2_eeg2_extracted)
x_train_2_eeg_stand = standardization(x_train_2_eeg)

x_train_3_eeg1_extracted = apply_fun(x_train_3_eeg1.reshape(21600, 512, 1))
x_train_3_eeg2_extracted = apply_fun(x_train_3_eeg2.reshape(21600, 512, 1))
x_train_3_eeg = concatenate_eeg(x_train_3_eeg1_extracted , x_train_3_eeg2_extracted)
x_train_3_eeg_stand = standardization(x_train_3_eeg)


# In[84]:


x_train_1_emg_stand = standardization(x_train_1_emg)
x_train_1 = concatenate_total(x_train_1_eeg_stand, x_train_1_emg_stand)

x_train_2_emg_stand = standardization(x_train_2_emg)
x_train_2 = concatenate_total(x_train_2_eeg_stand, x_train_2_emg_stand)

x_train_3_emg_stand = standardization(x_train_3_emg)
x_train_3 = concatenate_total(x_train_3_eeg_stand, x_train_3_emg_stand)


# In[ ]:


##################################################################


# In[95]:




nn_train_emg = np.concatenate((x_train_1,x_train_3),axis = 0)
label_total = np.concatenate((label_1, label_3), axis = 0)


# In[96]:


print(sum(label_total == 0))
print(sum(label_total == 1))
print(sum(label_total == 2))


y_pred_mat = np.zeros((label_2.shape[0], 5))

for k in range(5):
    np.random.seed(k)
    class0_idx = np.where(label_total == 0)[0]
    class1_idx = np.where(label_total == 1)[0]
    del_class0_idx = np.random.choice(class0_idx, size = len(class0_idx)-sum(label_total  == 2), replace=False) 
    del_class1_idx = np.random.choice(class1_idx, size = len(class1_idx)-sum(label_total  == 2), replace=False) 
    print('del_0_idx:', del_class0_idx[:3])
    print('del_1_idx:', del_class1_idx[:3])
    nn_train_deleted = np.delete(nn_train_emg, (np.hstack((del_class0_idx, del_class1_idx))), axis = 0) 
    label_total_deleted = np.delete(label_total, (np.hstack((del_class0_idx, del_class1_idx))), axis = 0) 





    label_total_trans = keras.utils.to_categorical(label_total_deleted, 3)






    # In[130]:



    nn = Sequential()
    nn.add(InputLayer((512,3)))
    # nn.add(Conv1D(100, 30, strides=1, activation='relu'))
    # nn.add(Dropout(0.3))
    nn.add(Conv1D(10, 5, strides=1, activation='relu'))
    nn.add(Dropout(0.3))
    nn.add(Conv1D(50, 10, strides=1, activation='relu'))
    nn.add(Dropout(0.3))
    nn.add(Conv1D(100, 20, strides=1, activation='relu'))
    nn.add(Dropout(0.3))
    nn.add(MaxPooling1D(2))
    nn.add(Conv1D(50, 30, strides=1, activation='relu'))
    nn.add(Dropout(0.3))
    nn.add(Conv1D(10, 30, strides=1, activation='relu'))
    nn.add(Dropout(0.3))
    nn.add(LSTM(5, activation ='relu', return_sequences= True, return_state= False))
    nn.add(Dropout(0.3))
    nn.add(Flatten())
    nn.add(Dense(500,activation = 'relu'))
    nn.add(Dropout(0.3))
    nn.add(Dense(200,activation = 'relu'))
    nn.add(Dropout(0.3))
    nn.add(Dense(50,activation = 'relu'))
    nn.add(Dropout(0.3))
    nn.add(Dense(3,activation = 'softmax'))
    optim = keras.optimizers.Adadelta()
    nn.compile(optimizer=optim,
              loss='categorical_crossentropy',
              metrics=['categorical_accuracy'])




    # In[ ]:


    nn.fit(nn_train_deleted, label_total_trans, epochs=200, verbose=2, batch_size = 200,callbacks=[EarlyStopping(monitor='loss', patience=8)])


    # In[118]:


    y_pred_mat[:,k] = np.argmax(nn.predict(x_train_2), axis=1) 


y_pred = np.zeros(y_pred_mat.shape[0])
for j in range(y_pred_mat.shape[0]):
    y_pred[j] =  Counter(y_pred_mat[j]).most_common(1)[0][0]

print('vote_mat:',y_pred_mat[:5])
print('vote_result:', y_pred[:5])


print('cv_score:', balanced_accuracy_score(label_2, y_pred))



# with open('output.csv', 'w') as f:
#     f.write("{},{}\n".format("Id", "y"))
#     for i in range(len(y_testid)):
#         f.write("{},{}\n".format(y_testid[i], results[i]))


# #


    

