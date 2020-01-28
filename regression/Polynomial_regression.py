import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def FitPolynomialRegression(K, x, y):
    rows = x.size
    poly = np.zeros([rows, K+1])
    for i in range(K+1):
        poly[:, i] = np.power(x, i)
    xTx = np.dot(np.transpose(poly), poly)
    x_plus = np.dot(np.linalg.inv(xTx),(np.transpose(poly)))
    W = np.dot(x_plus,y)
    return W


def EvalPolynomial(x, w):
    cols = w.size
    rows = x.size
    poly = np.zeros([rows,cols])
    for i in range(cols):
        poly[:,i] = np.power(x,i)
    y_hat = np.dot(poly, w)
    return y_hat


def getBestPolynomial(xTrain, yTrain, xTest, yTest, h):
    mse = float('inf')
    mse_training = np.zeros(h)
    mse_testing = np.zeros(h)
    for i in range(h):
        w = FitPolynomialRegression(i, xTrain, yTrain)
        y_hat_test = EvalPolynomial(xTest, w)
        y_hat_train = EvalPolynomial(xTrain, w)
        mse_testing[i] = ((sum(np.square(y_hat_test - yTest))))
        mse_training[i] = ((sum(np.square(y_hat_train - yTrain))))
    return np.array([mse_training, mse_testing]), np.argmin(mse_testing)


def plotResidual(mseTrain, mseTest, h):
    plt.plot(np.linspace(1, h, h), mseTest, c='blue')
    plt.plot(np.linspace(1, h, h), mseTrain, c='green')
    plt.gca().legend(("Testing", "Training"))
    plt.show()


if __name__ == '__main__':
    polyreg = pd.read_csv('polyreg.csv')
    XtrainData, XtestData, YtrainData, YtestData = train_test_split(polyreg['x'], polyreg['y'], test_size=0.25)
    h = 10
    residual, degree = getBestPolynomial(XtrainData,YtrainData,XtestData,YtestData,h)
    print(residual)
    print('The minimum degree is: ',degree)
    plotResidual(residual[0],residual[1], h)