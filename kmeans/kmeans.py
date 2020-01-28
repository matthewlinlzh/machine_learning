from numpy import *


def distance(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2)))


def randCent(dataSet, k):
    n = shape(dataSet)[1]
    centroids = mat(zeros((k,n))) #create centroid mat
    for j in range(n): #create random cluster centers, within bounds of each dimension
        minJ = min(dataSet[:,j])
        rangeJ = float(max(dataSet[:,j]) - minJ)
        centroids[:,j] = mat(minJ + rangeJ * random.rand(k,1))
    return centroids


def my_kmeans(feature, k):
    m = shape(feature)[0]
    centorids = randCent(feature, k)
    clusterCenter = mat(zeros((m,2)))
    centMove = True
    while centMove:
        centMove = False
        for i in range(m):
            minD = inf; minIndex = -1
            for j in range(k):
                dist = distance(centorids[j, :], feature[i, :])
                if dist < minD: minD = dist; minIndex = j
            if clusterCenter[i,0] != minIndex: centMove = True
            clusterCenter[i, :] = minIndex,minD**2
        for cent in range(k):
            ptsInClust = feature[nonzero(clusterCenter[:, 0].A == cent)[0]]
            centorids[cent, :] = mean(ptsInClust, axis=0)
    return centorids,clusterCenter


def my_kmeans_plot(clusters):
    # making plot
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set()

    label = array(clusterCenter)[:, 0]
    color = ['r', 'g', 'b','c', 'y']
    for i in range(dataMat.shape[1] - 1):
        plt.figure()
        for j in range(dataMat.shape[0]):
            plt.scatter(dataMat[j, i], dataMat[j, i+1], c=color[int(label[j])], s=30)
        plt.scatter(centorids[:, i], centorids[:, i+1], c='black', s=200, alpha=0.5)
    plt.show()


if __name__ == "__main__":
    dataMat = []
    fr = open('customer.csv')
    fr.readline()
    fr.readline()
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = curLine[0].split(',')
        del fltLine[0]
        if fltLine[0] == 'Male': fltLine[0] = 0
        else: fltLine[0] = 1
        for i in range(len(fltLine)): fltLine[i] = float(fltLine[i])
        dataMat.append(fltLine)
    dataMat = array(dataMat)[:,:3]
    centorids,clusterCenter = my_kmeans(dataMat, 5)
    centorids = array(centorids)
    my_kmeans_plot(centorids)

