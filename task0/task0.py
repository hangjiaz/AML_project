import argparse
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.covariance import MinCovDet as mcd


# read train file
traindata = pd.read_csv("train.csv", header=0)
x_train = traindata.iloc[:, 2:12].values
y_train = traindata.iloc[:, 1].values

# read test file
testdata = pd.read_csv("test.csv", header=0)
x_test = testdata.iloc[:, 1:11].values
y_testid = testdata.iloc[:, 0].values


# remove outlier
def detectOutliers(data):
    dist = mcd(random_state=0)
    dist.fit(x_train)
    MDsqr = dist.mahalanobis(data)
    MD = np.sqrt(MDsqr)
    std = np.std(MD)
    k = 2*std
    m = np.mean(MD)
    up_t = m + k
    low_t = m - k
    outliers = []
    for i in range(len(MD)):
        if (MD[i] >= up_t) or (MD[i] <= low_t):
            outliers = np.append(outliers, i)  # index of the outlier
    return outliers


index = detectOutliers(x_train)
for k in range(len(index) - 1, -1, -1):
    x_train = np.delete(x_train, index[k], axis=0)
    y_train = np.delete(y_train, index[k], axis=0)



# Train the model using training set
regr = LinearRegression()
regr.fit(x_train, y_train)

# predict on the test set
y_pred = regr.predict(x_test)

# create output file recording predict results

file_name = "output.csv"
f_out = open(file_name, 'w')


f_out.write("{}{}\n".format("Id,", "y"))

for i in range(len(y_pred)):
    f_out.write("{},{}\n".format(y_testid[i], y_pred[i]))

f_out.close()

# result evaluation

# Obtain true labels of test set
y_true = []
for i in range(len(x_test)):
    y_true.append(np.mean(x_test[i]))

# calculate RMSE
RMSE = mean_squared_error(y_true, y_pred)**0.5

print('Evaluation: Root Mean Squared Error is {}'.format(RMSE))
