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
#x_ktrain, x_ktest, y_ktrain, y_ktest = train_test_split(x_train, y_train, test_size=0.4, random_state=0)


################## downsampling

#class0_idx = np.where(y_ktrain == 0)[0]

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
#   X.append(np.sum(filtered_sig))
    
    X += list(np.median(temp, axis=0))
    X += list(np.min(temp, axis=0))
    X += list(np.max(temp, axis=0))
    X = np.array(X)
    

    return X

train_data = np.apply_along_axis(feature_extraction, 1, x_train)
test_data = np.apply_along_axis(feature_extraction, 1, x_test)

################################

scaler = StandardScaler()
x_train_stand = scaler.fit_transform(train_data)
x_test_stand = scaler.transform(test_data)

lr = LogisticRegression(random_state=0,penalty='l1').fit(x_train_stand, y_train)
model = SelectFromModel(lr, prefit=True)
x_train_selected = model.transform(x_train_stand)
x_test_selected   = model.transform(x_test_stand)







#selector = SelectKBest(f_classif, k=500)
#x_train_selected  = selector.fit_transform(x_train_stand, y_ktrain)
#x_test_selected = selector.transform(x_test_stand)


#y_pred_mat = np.zeros((y_testid.shape[0], 9))

#for k in range(9):
    
#np.random.seed(0)
#del_class0_idx = np.random.choice(class0_idx, size = len(class0_idx)-sum(y_ktrain == 2), replace=False) 
#print('del_class0_idx:', del_class0_idx[:5])
#x_ktrain = np.delete(x_ktrain, (del_class0_idx), axis = 0) 
#y_ktrain = np.delete(y_ktrain, (del_class0_idx), axis = 0) 
#print('downsampling shape:', sum(y_ktrain == 0), sum(y_ktrain == 1), sum(y_ktrain == 2), sum(y_ktrain == 3))


#
#score = f1_score(y_ktest, y_kpred, average='micro')
#print(score)


         
    
clf = LGBMClassifier(random_state=0, n_estimators=100, max_depth=20, num_leaves=50, class_weight = {0:0.2, 1:0.3, 2:0.2, 3:0.3}).fit(x_train_selected,y_train)
y_pred  = clf.predict(x_test_selected)
    
# #    y_pred_mat[:,k] = clf.predict(xgb.DMatrix(x_test_stand))
# score = f1_score(y_ktest, y_kpred, average='micro')
# print(score)


#y_pred = np.zeros(y_testid.shape[0])
#for j in range(y_pred_mat.shape[0]):
#    y_pred[j] =  Counter(y_pred_mat[j]).most_common(1)[0][0]
#
#print('vote_mat:',y_pred_mat[:5])
#print('vote_result:', y_pred[:5])


with open('output.csv', 'w') as f:
    f.write("{},{}\n".format("id", "y"))
    for i in range(len(y_testid)):
        f.write("{},{}\n".format(y_testid[i], y_pred[i]))



