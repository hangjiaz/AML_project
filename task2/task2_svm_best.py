import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from collections import Counter



############## read data

x_traindata = pd.read_csv("X_train.csv", header=0)
y_traindata = pd.read_csv("y_train.csv", header=0)
testdata = pd.read_csv("X_test.csv", header=0)   	

x_train = x_traindata.iloc[:,1:1001].values
y_train = y_traindata.iloc[:,1].values
x_test = testdata.iloc[:,1:1001].values

y_testid = testdata.iloc[:,0].values



class1_idx = np.where(y_train == 1)[0]

y_pred_mat = np.zeros((y_testid.shape[0], 9))


for k in range(9):


	x_train = x_traindata.iloc[:,1:1001].values
	y_train = y_traindata.iloc[:,1].values
	x_test = testdata.iloc[:,1:1001].values


	############ downsampling
	np.random.seed(k)
	del_class1_idx = np.random.choice(class1_idx, size = len(class1_idx)-sum(y_train == 0), replace=False) 
	print('del_class1_idx:', del_class1_idx[:5])
	x_train = np.delete(x_train, (del_class1_idx), axis = 0) 
	y_train = np.delete(y_train, (del_class1_idx), axis = 0) 


    ############   data standardization 
	scaler = StandardScaler()
	x_train_norm = scaler.fit_transform(x_train)
	x_test_norm = scaler.transform(x_test)

	############   feature transformation

	lr = LogisticRegression(random_state=0,penalty='l2').fit(x_train_norm, y_train)
	model = SelectFromModel(lr, prefit=True)
	x_train_norm  = model.transform(x_train_norm)
	x_test_norm  = model.transform(x_test_norm)


# ############   remove outliers
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



# score = 0
# for i in range(5):
#     x_ktrain, x_ktest, y_ktrain, y_ktest = train_test_split(x_train_norm, y_train, test_size=0.4, random_state=i)
#     clf = SVC(C = 3, kernel = 'rbf',decision_function_shape ='ovr', gamma = 'auto', class_weight={0: 0.45, 1: 0.1, 2: 0.45})
#     clf.fit(x_ktrain, y_ktrain) 
#     y_kpred = clf.predict(x_ktest)
#     score += balanced_accuracy_score(y_ktest, y_kpred)
# score /= 5

# print(score)

	clf = SVC(C = 3, kernel = 'rbf',decision_function_shape ='ovr', gamma = 'auto')
	clf.fit(x_clean, y_clean)
	y_pred_mat[:,k]= clf.predict(x_test_norm)

y_pred = np.zeros(y_testid.shape[0])
for j in range(y_pred_mat.shape[0]):
    y_pred[j] =  Counter(y_pred_mat[j]).most_common(1)[0][0]

print('mat:',y_pred_mat[:5])
print('pred:',y_pred[:5])

 # ############## write output files
with open('output.csv', 'w') as f:
   f.write("{},{}\n".format("id", "y"))
   for i in range(len(y_testid)):
   		f.write("{},{}\n".format(y_testid[i], y_pred[i]))




 
