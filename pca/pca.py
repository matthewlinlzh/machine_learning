from numpy import *
import matplotlib.pyplot as plt


def my_pca(dataMat, k):
    meanVec = dataMat - mean(dataMat, axis=0)
    covMat = cov(meanVec, rowvar=0)
    eigenVal, eigenVec = linalg.eig(covMat)
    lowD = dot(meanVec, eigenVec[:,:k])
    return lowD


def my_pca_plot(lowDimMat):
    m, n = low_dimension_Mat.shape
    for i in range(n - 1):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for j in range(m):
            ax.scatter(lowDimMat[j, i], lowDimMat[j ,i+1], marker='o', s=50, c='red')
    plt.show()


if __name__ == "__main__":
    from sklearn import datasets
    iris = datasets.load_iris()
    x = iris.data
    low_dimension_Mat = my_pca(x, 2)
    print(low_dimension_Mat)
    low_dimension_Mat = my_pca(x, 3)
    print(low_dimension_Mat)
    my_pca_plot(low_dimension_Mat)

