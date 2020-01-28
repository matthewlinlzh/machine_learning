import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn import svm
import seaborn
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report,confusion_matrix



#Load the dataseta and convert it to a dataframe 
# label= target
cancer = load_breast_cancer()
df_cancer= pd.DataFrame(np.c_[cancer['data'],cancer['target']],columns=np.append(cancer['feature_names'],['target']))


#################################
# Visualize the relationship between features
# use seaborn.pairplot

# Add code here function data_vis(df_cancer)
def data_vis(df_cancer):
    # seaborn.pairplot(df_cancer)
    # plt.show()
    seaborn.set(style='ticks')
    seaborn.pairplot(df_cancer, vars=['mean radius', 'mean texture', 'mean perimeter', 'mean area','mean smoothness'], hue='target')
    plt.show()


#################################

# check correlation beteeen all features 
# use seaborn.heatmap

# Add code here function data_h_map(df_cancer)
def data_h_map(df_cancer):
    plt.figure(figsize=(20,12))
    seaborn.heatmap(df_cancer.corr(), annot=True)
    plt.show()

#################################

# Use linear svm for binary classification

# Add code here lin_svm(df_cancer)
	# 1- split train and test  (80%,20%)
	# 2- fit svm 
	# 3- test svm
	# 4- Print classification report
    # Label x and y variables from our dataset

def lin_svm(df_cancer):
    # feature = df_cancer.columns[0:30]
    x = df_cancer.drop(['target'], axis=1)
    y = df_cancer['target']
    xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2)
    svmClassifier = svm.SVC(gamma='auto', kernel='linear')
    svmClassifier.fit(xTrain,yTrain)
    y_hat = svmClassifier.predict(xTest)
    return classification_report(yTest,y_hat), confusion_matrix(yTest, y_hat)


if __name__ == '__main__':
    data_vis(df_cancer)
    data_h_map(df_cancer)
    svmReport, conMat = lin_svm(df_cancer)
    print(svmReport, '\n', conMat)

