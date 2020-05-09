"""
Homework: Principal Component Analysis
Course  : Data Mining II (636-0019-00L)
"""

#import all necessary functions


from helper_function import *
from sklearn.feature_selection import VarianceThreshold
from scipy.spatial import distance
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
'''
Main Function
'''
if __name__ in "__main__":
    #Initialise plotting defaults
    initPlotLib()

    ##################
    #Exercise 2:
    
    #Simulate Data
    # read data file
    x_traindata = pd.read_csv("X_train.csv", header=0)
    y_traindata = pd.read_csv("y_train.csv", header=0)
    x_train = x_traindata.iloc[:, 1:833].values
    y_train = y_traindata.iloc[:, 1].values


    # mean imputation
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    x_train_imputed = imp_mean.fit_transform(x_train)

    
    #Perform a PCA
    #1. Compute covariance matrix
    selector = VarianceThreshold(0)
    x_train_imputed = selector.fit_transform(x_train_imputed)
    normalised_data = dataNormalisation(x_train_imputed )

    cov = computeCov(normalised_data)
    #2. Compute PCA by computing eigen values and eigen vectors
    [eigen_values,eigen_vectors] = computePCA(cov)    
    #3. Transform your input data onto a 2-dimensional subspace using the first two PCs
    transformed_data = transformData(eigen_vectors[:,0:2],normalised_data)
    #4. Plot your transformed data and highlight the three different sample classes
    y_class  = classify_y(y_train)
    plotTransformedData(transformed_data, y_class,"mean_shape.pdf")
    #5. How much variance can be explained with each principle component?
    var = computeVarianceExplained(eigen_values)
    sp.set_printoptions(precision=2)
    print("Variance Explained: ")
    for i in range(15):
        print("PC %d: %.2f"%(i+1,var[i]))
    #6. Plot cumulative variance explained per PC
    plotCumSumVariance(var,"cumvar.pdf")
    
    #7 delete outlier based on PCA distance

           
       
    
    