import numpy as np 
import pandas as pd
from sklearn import decomposition

from numpy import array, dot, mean, std, empty, argsort
from numpy.linalg import eigh, solve
from numpy.random import randn
from mpl_toolkits.mplot3d import Axes3D
from numpy import *
import matplotlib.pylab as plt






#read train data in panda data frame
data_set = pd.read_csv("./train.csv" , index_col = 0  )
#extract features and labels
data_set = data_set[0:400]
features_columns = list(data_set)[1:]
features = np.asarray(data_set[features_columns])

def cov(data):
    """
        covariance matrix
        note: specifically for mean-centered data
    """
    N = data.shape[1]
    C = empty((N, N))
    for j in range(N):
        C[j, j] = mean(data[:,j] * data[:, j])
        for k in range(N):
            C[j, k] = C[k, j] = mean(data[0:, j] * data[0:, k])
    return C

def pca(data, pc_count = None):
    """
        Principal component analysis using eigenvalues
        note: this mean-centers and auto-scales the data (in-place)
    """
    data -= mean(data, 0)
    data /= std(data, 0)
    C = cov(data)
    E, V = eigh(C)
    key = argsort(E)[::-1][:pc_count]
    E, V = E[key], V[:, key]
    U = dot(V.T, data.T).T
    return U, E, V


""" visualize """
#first we need to map colors on labels

dfcolor = pd.DataFrame([[0,'red'],[1,'blue'],[2,'yellow']],columns=['y','Color'])
tran = pca(features,pc_count= 2)[0]
data_set['pca1'] = tran[:,0]
data_set['pca2'] = tran[:,1]
#data_set['pca3'] = tran[:,2]
dataframe = pd.merge(data_set,dfcolor)


 
 
# Create plot
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, axisbg="1.0")

 
for x,y ,color, labels in zip(dataframe['pca1'],dataframe['pca2'], dataframe['Color'], dataframe['y']):
    ax.scatter(x, y, alpha=0.8, c=color, edgecolors='none', s=30)
 
plt.title('Matplot 3d scatter plot')
plt.legend(loc=2)
plt.show()






