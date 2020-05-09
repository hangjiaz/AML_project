import pandas as pd
import scipy as sp
import numpy as np
from sklearn.preprocessing import StandardScaler, normalize
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from sklearn.covariance import MinCovDet as mcd
from sklearn.metrics import balanced_accuracy_score
import lightgbm as lgb
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import xgboost as xgb
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.linear_model import LogisticRegression


def mlp_1(x_train, y_train, x_val, y_val, x_test, testid):
    # data standardization

    #y_transform
    y_train_transformed = keras.utils.to_categorical(y_train, 3)

    model = Sequential()
    model.name = 'model'
    model.add(Dense(200, activation='relu', kernel_initializer='random_uniform', input_shape=(x_train.shape[1],)))
    model.add(Dense(3, activation='softmax',kernel_initializer='random_uniform'))

    #optim = keras.optimizers.sgd(lr=0.01,decay = 1e-4,momentum=0.9)
    optim = keras.optimizers.Adadelta()
    #optim = keras.optimizers.RMSprop()
    model.compile(optimizer=optim,
              loss='categorical_crossentropy',
              metrics=['categorical_accuracy'])
    #from sklearn.model_selection import train_test_split
    #X_train_, X_test, y_train_, y_test_ = train_test_split(x_stand_train, y_train,test_size=0.33)
    model.fit(x_train, y_train_transformed, batch_size=100, epochs=45, verbose=1)
    y_pred_val = sp.argmax(model.predict(x_val), axis = 1)
    BMAC = balanced_accuracy_score(y_val, y_pred_val)
    print("BMAC Score is ", BMAC)
    y_pred = sp.argmax(model.predict(x_test), axis=1)

    return y_pred, testid

def mlp_2(x_train, y_train, x_val, y_val, x_test, testid):
    # data standardization

    #y_transform
    y_train_transformed = keras.utils.to_categorical(y_train, 3)

    model = Sequential()
    model.name = 'model'
    model.add(Dense(1500, activation='relu', kernel_initializer='random_uniform', input_shape=(x_train.shape[1],)))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Dense(1500, activation='relu', kernel_initializer='random_uniform', input_shape=(1500,)))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Dense(150, activation='relu', kernel_initializer='random_uniform', input_shape=(1500,)))
    model.add(Dropout(0.4))
    model.add(BatchNormalization())
    model.add(Dense(20, activation='relu', kernel_initializer='random_uniform', input_shape=(150,)))
    model.add(Dropout(0.1))
    model.add(BatchNormalization())
    model.add(Dense(3, activation='softmax',kernel_initializer='random_uniform'))

    #optim = keras.optimizers.sgd(lr=0.01,decay = 1e-4,momentum=0.9)
    optim = keras.optimizers.Adadelta()
    #optim = keras.optimizers.RMSprop()
    model.compile(optimizer=optim,
              loss='categorical_crossentropy',
              metrics=['categorical_accuracy'])
    #from sklearn.model_selection import train_test_split
    #X_train_, X_test, y_train_, y_test_ = train_test_split(x_stand_train, y_train,test_size=0.33)
    model.fit(x_train, y_train_transformed, batch_size=100, epochs=100, verbose=1)
    y_pred_val = sp.argmax(model.predict(x_val), axis = 1)
    BMAC = balanced_accuracy_score(y_val, y_pred_val)
    print("BMAC Score is ", BMAC)
    y_pred = sp.argmax(model.predict(x_test), axis=1)

    return y_pred, testid


def light_gbm(x_train, y_train, x_val, y_val, x_test, testid):
    scaler = StandardScaler()
    x_stand_train = scaler.fit_transform(x_train)
    x_stand_val = scaler.transform(x_val)
    x_stand_test = scaler.transform(x_test) 

    lgb_train = lgb.Dataset(x_stand_train, y_train)
    lgb_eval = lgb.Dataset(x_stand_val, y_val, reference=lgb_train)

    params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'multiclassova',
    'is_unbalance': False,    
    #'metric': {'l1', 'l2'},
    'num_class': 3, 
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.5,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
    }   

    gbm = lgb.train(params,lgb_train, num_boost_round=500, feval=custom_accuracy, valid_sets={lgb_train, lgb_eval}, early_stopping_rounds=20)
    y_pred = gbm.predict(x_test, num_iteration=gbm.best_iteration)

    return y_pred, testid

def custom_accuracy(preds, train_data):
    labels = train_data.get_label()
    n = len(labels)
    results = []
    for i in range(n):
        results.append(np.argmax([preds[i], preds[n + i], preds[2*n + i]]))
    return 'BMAC', balanced_accuracy_score(labels, results), True

def model_zoo(x_train,y_train,x_val,y_val,x_test,testid):
    #this function is for selecting models
    # possible models includes: nusvc, GradientBoostingClassifier, and AdaBoostClassifier
    scaler = StandardScaler()
    x_stand_train = scaler.fit_transform(x_train)
    x_stand_val = scaler.transform(x_val)
    x_stand_test = scaler.transform(x_test) 

    classifiers = [ KNeighborsClassifier(30), SVC(kernel="rbf", probability=True, decision_function_shape="ovo"), 
                LinearSVC(C=2.5 ,class_weight="balanced"), 
                RandomForestClassifier(n_estimators=150), 
                GradientBoostingClassifier(n_estimators=150)]

    for clf in classifiers: 
        clf.fit(x_train, y_train) 
        name = clf.__class__.__name__ 
        print("="*30,'\n', name)
    
        y_pred_val = clf.predict(x_val) 
        BMAC = balanced_accuracy_score(y_val, y_pred_val) 
        print("BMAC of this model: ", BMAC)
        print("\n")
        print("="*30)

    return

def nusvc_model(x_train, y_train, x_val, y_val, x_test, testid):
    scaler = StandardScaler()
    x_stand_train = scaler.fit_transform(x_train)
    x_stand_val = scaler.transform(x_val)
    x_stand_test = scaler.transform(x_test) 

    #nus = [0.05,0.1,0.2,0.3,0.4,0.5,0.6]
    #for i in range(len(nus)):
    clf = NuSVC(nu = 0.3, kernel="linear", probability=True, decision_function_shape="ovo", gamma="scale", class_weight="balanced")
    clf.fit(x_train,y_train)

    y_pred_val = clf.predict(x_val) 
    BMAC = balanced_accuracy_score(y_val, y_pred_val) 
    print("BMAC of this model: ", BMAC)
    print("\n")
    print("="*30)

    y_pred = clf.predict(x_test)

    return y_pred,testid

def svc_model(x_train, y_train, x_test, testid): 
    #params={'C':[0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6]}
    #for i in range(len(nus)):
    #acc_scorer = make_scorer(accuracy_score)
    clf = SVC(C = 3.5, kernel="rbf", probability=True, decision_function_shape="ovo", gamma="scale", class_weight="balanced")
    #grid_obj = GridSearchCV(clf, parameters, scoring=acc_scorer)
    #grid_obj = grid_obj.fit(X_train, y_train)

    # Set the clf to the best combination of parameters

    # Fit the best algorithm to the data. 
    clf.fit(x_train, y_train)
    #clf.fit(X_train, )
    #weight = clf.coef_
    #weight = normalize(weight)


    #print(np.where(weight >= 0.1))

    #y_pred_val = clf.predict(x_val) 
    #BMAC = balanced_accuracy_score(y_val, y_pred_val) 
    #print("BMAC of this model: ", BMAC)
    #print("\n")
    #print("="*30)

    y_pred = clf.predict(x_test)

    return y_pred,testid

def linearsvc_model(x_train, y_train, x_val, y_val, x_test, testid):
    scaler = StandardScaler()
    x_stand_train = scaler.fit_transform(x_train)
    x_stand_val = scaler.transform(x_val)
    x_stand_test = scaler.transform(x_test) 

    #nus = [0.05,0.1,0.2,0.3,0.4,0.5,0.6]
    #for i in range(len(nus)):
    clf = LinearSVC(C=2 ,class_weight="balanced")
    clf.fit(x_train,y_train)

    y_pred_val = clf.predict(x_val) 
    BMAC = balanced_accuracy_score(y_val, y_pred_val) 
    print("BMAC of this model: ", BMAC)
    print("\n")
    print("="*30)

    y_pred = clf.predict(x_test)

    return y_pred,testid

def mixture_model(x_train, y_train, x_val, y_val, x_test, testid):
    #nus = [0.05,0.1,0.2,0.3,0.4,0.5,0.6]
    #for i in range(len(nus)):
    clf1 = SVC(C = 1, kernel="rbf", probability=True, decision_function_shape="ovo", gamma="scale", class_weight="balanced")
    clf2 = NuSVC(kernel="rbf", probability=True, decision_function_shape="ovo")
    clf3 = GradientBoostingClassifier()
    clf4 = RandomForestClassifier()

    vote = VotingClassifier(estimators=[('svc', clf1), ('nusvc', clf2)], voting='soft')


    vote.fit(x_train,y_train)

    y_pred_val = vote.predict(x_val) 
    BMAC = balanced_accuracy_score(y_val, y_pred_val) 
    print("BMAC of this model: ", BMAC)
    print("\n")
    print("="*30)

    y_pred = vote.predict(x_test)

    return y_pred,testid


def xgb_model(x_train, y_train, x_val, y_val, x_test, testid): 

    #data_dmatrix = xgb.DMatrix(data=x_stand_train,label=y_train)

    #nus = [0.05,0.1,0.2,0.3,0.4,0.5,0.6]
    #for i in range(len(nus)):
    xg_class = xgb.XGBClassifier(objective ='multi:softmax', colsample_bytree = 0.3, learning_rate = 0.01,
                max_depth = 3, alpha = 10, n_estimators = 40)
    xg_class.fit(x_train,y_train)

    y_pred_val = xg_class.predict(x_val) 
    BMAC = balanced_accuracy_score(y_val, y_pred_val) 
    print("BMAC of this model: ", BMAC)
    print("\n")
    print("="*30)

    y_pred = xg_class.predict(x_test)

    return y_pred,testid











