import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split


def RBFregression(xtrain, ytrain, xtest, ytest):
    Rbf = KernelRidge(kernel='rbf', alpha=0.5)
    Rbf.fit(xtrain, ytrain)
    yPredit = Rbf.predict(xtest)
    accuracy = np.sqrt(np.abs(1 - (sum(np.square(ytest - yPredit)))/(sum(np.square(ytest-np.mean(yPredit))))))
    return accuracy


if __name__ == "__main__":
    boston_dataset = load_boston()
    boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
    X = boston[['RM', 'AGE', 'DIS', 'RAD', 'TAX']]  # take 5 features
    y = boston_dataset.target  # target value
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    accuracy = RBFregression(x_train, y_train, x_test, y_test)
    print(accuracy)