import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def sigmoid(x):
    x = x.astype(np.float128)
    return 1/(1+np.exp(-x))


def loss(h,y):
    epsilon = 1e-10
    return np.mean(-y * np.log(h + epsilon)- (1-y) * np.log(1-h + epsilon))


def logisticFit(inputData, labelData, alpha, tolerance):
    weight = np.zeros(inputData.shape[1])
    diff_loss = tolerance * 2
    preLoss = float('inf')
    max_iter = 10000
    for i in range(max_iter):
        if diff_loss < tolerance:
            return weight,i
        else:
            z = np.dot(inputData, weight).astype(np.float128)
            h = sigmoid(z)
            gradient = np.transpose(inputData).dot(h - labelData)
            weight = weight - alpha * gradient
            newLoss = loss(h, labelData)
            diff_loss = np.abs(newLoss - preLoss)
            preLoss = newLoss
    return weight, max_iter


def findScore(predict, testLabel):
    True_positive = sum(predict * testLabel)
    Recall = True_positive / sum(testLabel)
    Precision = True_positive / sum(predict)
    F1Score = 2 * Recall * Precision / (Recall + Precision)
    return Recall, Precision, F1Score


if __name__ == "__main__":
    # take the first two classes of the dataset i.e., first 100 instances
    iris = datasets.load_iris()
    x = iris.data[:100, :]
    x = np.hstack([np.ones([x.shape[0],1]), x])
    Y = iris.target[:100]  # the labels
    print(type(iris))
    # xtrain, xtest, ytrain, ytest = train_test_split(x, Y, test_size=0.2)
    # weight, iter= logisticFit(xtrain,ytrain,0.5, 0.9)
    # z = np.dot(xtest, weight)
    # predict_prob = sigmoid(z)
    # predict = []
    # for i in predict_prob:
    #     if (i >= 0.5):
    #         predict.append(1)
    #     else:
    #         predict.append(0)
    # R, P, F1 = findScore(predict, ytest)
    # print(iter)
    # print(R)
    # print(P)
    # print(F1)
    #
    #
    # fig = plt.figure()
    # x1 = xtest[:,1]
    # x2 = xtest[:,2]
    # x = [np.min(x1 - 1), np.max(x2 + 3)]
    # y = - (weight[0] + np.dot(weight[1], x)) / weight[2]
    # plt.plot(x, y, label='Decision Boundary')
    # plt.scatter(x1, x2 ,c=ytest, cmap="RdBu", vmin=-.2, vmax=1.2, edgecolor="white", linewidth=1)
    # plt.legend()
    # plt.show()
#