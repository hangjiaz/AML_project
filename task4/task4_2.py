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
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso
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


# In[6]:


x_test_1_eeg1,  x_test_1_eeg2, x_test_1_emg  = read_data(1, 'test')
x_test_2_eeg1,  x_test_2_eeg2, x_test_2_emg  = read_data(2, 'test')


# In[7]:


label_1 = y_train[0:21600] 
label_2 = y_train[21600:43200]
label_3 = y_train[43200:64800]


# In[8]:



def eeg_feature_extraction(x):
    X = []
    [ts, filtered_sig, features_ts,theta,alpha_low, alpha_high,beta, gamma, plf_pairs, plf]  = eeg.eeg(signal=x, sampling_rate=128.0, show=False)    
    X.append(np.mean(theta))
    X.append(np.min(theta))
    X.append(np.max(theta))
    X.append(np.std(theta))
    X.append(np.mean(alpha_low))
    X.append(np.min(alpha_low))
    X.append(np.max(alpha_low))
    X.append(np.std(alpha_low))
    X.append(np.mean(alpha_high))
    X.append(np.min(alpha_high))
    X.append(np.max(alpha_high))
    X.append(np.std(alpha_high))
    X.append(np.mean(beta))
    X.append(np.min(beta))
    X.append(np.max(beta))
    X.append(np.std(beta))
    X.append(np.mean(gamma))
    X.append(np.min(gamma))
    X.append(np.max(gamma))
    X.append(np.std(gamma))
    X.append(np.mean(plf))
    X.append(np.min(plf))
    X.append(np.max(plf))
    X.append(np.std(plf))
    X.append(np.mean(filtered_sig))
    X.append(np.std(filtered_sig))
    X = np.array(X)
        
    return X



def apply_fun(x):
    result = []
    for i in range(x.shape[0]):
        result.append(eeg_feature_extraction(x[i]))
    result = np.array(result)
    return result


# In[9]:


x_train_1_eeg = np.hstack((x_train_1_eeg1,  x_train_1_eeg2)).reshape((21600,512,2))
x_train_1_eeg_extracted = apply_fun(x_train_1_eeg)
x_train_2_eeg = np.hstack((x_train_2_eeg1,  x_train_2_eeg2)).reshape((21600,512,2))
x_train_2_eeg_extracted = apply_fun(x_train_2_eeg)
x_train_3_eeg = np.hstack((x_train_3_eeg1,  x_train_3_eeg2)).reshape((21600,512,2))
x_train_3_eeg_extracted = apply_fun(x_train_3_eeg)


# In[10]:


x_test_1_eeg = np.hstack((x_test_1_eeg1,  x_test_1_eeg2)).reshape((21600,512,2))
x_test_1_eeg_extracted = apply_fun(x_test_1_eeg)
x_test_2_eeg = np.hstack((x_test_2_eeg1,  x_test_2_eeg2)).reshape((21600,512,2))
x_test_2_eeg_extracted = apply_fun(x_test_2_eeg)


# In[11]:


scaler = StandardScaler()
x_train_1_eeg_stand= scaler.fit_transform(x_train_1_eeg_extracted)
x_train_2_eeg_stand = scaler.fit_transform(x_train_2_eeg_extracted)
x_train_3_eeg_stand = scaler.fit_transform(x_train_3_eeg_extracted)
x_test_1_eeg_stand= scaler.fit_transform(x_test_1_eeg_extracted)
x_test_2_eeg_stand = scaler.fit_transform(x_test_2_eeg_extracted)


# In[108]:


#train_eeg = np.concatenate((x_train_1_eeg_stand,x_train_3_eeg_stand),axis = 0)
train_eeg = np.concatenate((x_train_1_eeg_stand,x_train_2_eeg_stand,x_train_3_eeg_stand),axis = 0)
test_eeg = np.concatenate((x_test_1_eeg_stand,x_test_2_eeg_stand),axis = 0)


# In[109]:


ifnrem_1 = (y_train[0:21600] == 2).astype(int)
ifnrem_2 = (y_train[21600:43200] == 2).astype(int)
ifnrem_3 = (y_train[43200:64800] == 2).astype(int)


# In[110]:


#ifnrem_train =  np.concatenate((ifnrem_1,ifnrem_3), axis = 0)
ifnrem_train =  np.concatenate((ifnrem_1,ifnrem_2,ifnrem_3), axis = 0)


# In[111]:


print(sum(ifnrem_train == 1))
print(sum(ifnrem_train == 0))
print(train_eeg.shape)


# In[142]:


print(train_eeg_deleted.shape)
print(ifnrem_train_deleted.shape)


# In[143]:



clf = SVC(C = 3, kernel = 'rbf',decision_function_shape ='ovr', gamma = 'scale')
clf.fit(train_eeg, ifnrem_train)
y_pred_ifnrem = clf.predict(test_eeg)


# In[17]:



#balanced_accuracy_score(ifnrem_2, y_pred_ifnrem)


# In[144]:


test_nrem_idx = np.where(y_pred_ifnrem == 1)
test_other_idx = np.where(y_pred_ifnrem == 0)


# In[ ]:


##################################################################


# In[114]:


label_1_13_idx = np.where(label_1 != 2 )
label_2_13_idx = np.where(label_2 != 2 )
label_3_13_idx = np.where(label_3 != 2 )


# In[115]:


x_train_1_emg_stand = standardization(x_train_1_emg)
x_train_2_emg_stand = standardization(x_train_2_emg)
x_train_3_emg_stand = standardization(x_train_3_emg)


# In[116]:


x_train_1_emg_13 = x_train_1_emg_stand[label_1_13_idx[0]]
x_train_2_emg_13 = x_train_2_emg_stand[label_2_13_idx[0]]
x_train_3_emg_13 = x_train_3_emg_stand[label_3_13_idx[0]]
#x_train_2_emg_13 = x_train_2_emg_stand[test_other_idx[0]]


# In[145]:


x_test_1_emg_stand = standardization(x_test_1_emg)
x_test_2_emg_stand = standardization(x_test_2_emg)
x_test_emg_stand = np.concatenate((x_test_1_emg_stand , x_test_2_emg_stand), axis = 0)
x_test_emg_13 = x_test_emg_stand[test_other_idx[0]]


# In[118]:


label_1_13 = label_1[label_1_13_idx[0]]
label_2_13 = label_2[label_2_13_idx[0]]
label_3_13 = label_3[label_3_13_idx[0]]
#label_2_13 = label_2[test_other_idx[0]]


# In[119]:


ifweak_1_13 = (label_1_13 == 1).astype(int)
ifweak_2_13 = (label_2_13 == 1).astype(int)
ifweak_3_13 = (label_3_13 == 1).astype(int)


# In[120]:



# nn_train_emg = np.concatenate((x_train_1_emg_13, x_train_3_emg_13),axis = 0)
# ifweak_train_13 = np.concatenate((ifweak_1_13,ifweak_3_13),axis = 0)
nn_train_emg = np.concatenate((x_train_1_emg_13,x_train_2_emg_13,x_train_3_emg_13),axis = 0)
ifweak_train_13 = np.concatenate((ifweak_1_13,ifweak_2_13,ifweak_3_13),axis = 0)


# In[146]:


print(sum(ifweak_train_13 == 1))
print(sum(ifweak_train_13 == 0))


# In[122]:


np.random.seed(1)
class1_idx = np.where(ifweak_train_13 == 1)[0]
del_class1_idx = np.random.choice(class1_idx, size = len(class1_idx)-sum(ifweak_train_13 == 0), replace=False) 
nn_train_emg_deleted = np.delete(nn_train_emg, (del_class1_idx), axis = 0) 
ifweak_train_13_deleted = np.delete(ifweak_train_13, (del_class1_idx), axis = 0) 


# In[123]:


ifweak_train_13_trans = keras.utils.to_categorical(ifweak_train_13_deleted, 2)


# In[147]:


print(ifweak_train_13_trans.shape)
#print(x_test_emg_13.shape)


# In[133]:



nn = Sequential()
nn.add(InputLayer((512,)))
nn.add(Reshape((512,1)))
# nn.add(Conv1D(100, 30, strides=1, activation='relu'))
# nn.add(Dropout(0.3))
nn.add(Conv1D(1, 13, strides=1, activation='relu'))
nn.add(Dropout(0.3))
nn.add(MaxPooling1D(2))
nn.add(LSTM(1, activation ='relu', return_sequences= True, return_state= False))
nn.add(Dropout(0.3))
nn.add(Flatten())
nn.add(Dense(100,activation = 'relu'))
nn.add(Dropout(0.3))
nn.add(Dense(50,activation = 'relu'))
nn.add(Dropout(0.3))
nn.add(Dense(2,activation = 'softmax'))
optim = keras.optimizers.Adadelta()
nn.compile(optimizer=optim,
          loss='categorical_crossentropy',
          metrics=['categorical_accuracy'])

nn.summary()


# In[134]:


nn.fit(nn_train_emg_deleted, ifweak_train_13_trans, epochs=200, verbose=2, batch_size = 200,callbacks=[EarlyStopping(monitor='loss', patience=10)])


# In[148]:


y_pred_ifweak = np.argmax(nn.predict(x_test_emg_13), axis=1)


# In[102]:


#balanced_accuracy_score(ifweak_2_13, y_pred_ifweak)


# In[149]:


test_weak_idx = np.where(y_pred_ifweak == 1)
test_rem_idx = np.where(y_pred_ifweak == 0)
test_weak_idx_original = test_other_idx[0][test_weak_idx[0]]
test_rem_idx_original = test_other_idx[0][test_rem_idx[0]]


# In[150]:


results = np.zeros(y_testid.shape[0])
#results = np.zeros(y_testid.shape[0])


# In[151]:


results[test_nrem_idx[0]]=2
results[test_weak_idx_original]=1
results[test_rem_idx_original]=3


# In[152]:


if sum(results == 0) == 0:
    print('ok')
#    print(balanced_accuracy_score(label_2, results))


# In[ ]:


##################### write to file ######################################


# In[153]:


with open('output.csv', 'w') as f:
    f.write("{},{}\n".format("Id", "y"))
    for i in range(len(y_testid)):
        f.write("{},{}\n".format(y_testid[i], results[i]))


# In[ ]:



#######################  visualization feature  ######################################################


# In[ ]:


class1_emg = x_train_emg[class1_idx[0][1]]
class2_emg = x_train_emg[class2_idx[0][1]]
class3_emg = x_train_emg[class3_idx[0][1]]


plt.figure(figsize=(10, 3))
plt.subplot(131)
plt.plot(class1_emg)
plt.subplot(132)
plt.plot(class2_emg)
plt.subplot(133)
plt.plot(class3_emg)
plt.suptitle('emg')
plt.show()


# In[ ]:


class1_eeg1 = x_train_eeg1[class1_idx[0][2],:].reshape((512,1))
class1_eeg2 = x_train_eeg2[class1_idx[0][2],:].reshape((512,1))
class1_eeg = np.hstack((class1_eeg1, class1_eeg2))
class2_eeg1 = x_train_eeg1[class2_idx[0][2],:].reshape((512,1))
class2_eeg2 = x_train_eeg2[class2_idx[0][2],:].reshape((512,1))
class2_eeg = np.hstack((class2_eeg1, class2_eeg2))
class3_eeg1 = x_train_eeg1[class3_idx[0][2],:].reshape((512,1))
class3_eeg2 = x_train_eeg2[class3_idx[0][2],:].reshape((512,1))
class3_eeg = np.hstack((class3_eeg1, class3_eeg2))

def feature_egg(x):
    [ts, filtered_sig, features_ts,theta,alpha_low, alpha_high,beta, gamma, plf_pairs, plf]  = eeg.eeg(signal=x, sampling_rate=128.0, show=False)    
    return ts, filtered_sig, features_ts,theta,alpha_low, alpha_high,beta, gamma, plf_pairs, plf
    
def plot_class(x1, x2, x3):
    [ts_1, filtered_sig_1, features_ts_1,theta_1,alpha_low_1, alpha_high_1,beta_1, gamma_1, plf_pairs_1, plf_1] = feature_egg(x1)  
    [ts_2, filtered_sig_2, features_ts_2,theta_2,alpha_low_2, alpha_high_2,beta_2, gamma_2, plf_pairs_2, plf_2] = feature_egg(x2)  
    [ts_3, filtered_sig_3, features_ts_3,theta_3,alpha_low_3, alpha_high_3,beta_3, gamma_3, plf_pairs_3, plf_3] = feature_egg(x3) 
    
   
    plt.figure(figsize=(10, 3))
    plt.subplot(131)
    plt.plot(features_ts_1,plf_1)
    plt.subplot(132)
    plt.plot(features_ts_2,plf_2)
    plt.subplot(133)
    plt.plot(features_ts_3,plf_3)
    plt.suptitle('plf')
    plt.show()


# In[ ]:





# In[ ]:





# In[ ]:








    

