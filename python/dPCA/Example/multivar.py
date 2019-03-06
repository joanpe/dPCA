#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 18:26:49 2019

@author: joan
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
PATH = '/home/joan/dPCA/python/dPCA'
sys.path.insert(0, PATH)
#from sklearn.decomposition import dPCA
from dPCA import dPCA


# Definition of the correlation matrix between the 3 neurons and means
a = np.random.randint(10, size=(3, 3))  # random matrix
np.fill_diagonal(a, 1)
corr = np.tril(a) + np.tril(a, -1).T

mean1 = [0, 0, 0]
mean2 = [2, 1, 10]
means = [mean1, mean2]

# Getting the data of the stimuli from a multivariate distribution with 
# mean = means and covariance = corr
n = 100
x1 = np.random.multivariate_normal(means[0], corr, size=n)
x2 = np.random.multivariate_normal(means[1], corr, size=n)
# X matrix has dimentions (n samples, 3 neurons, 2 stimuli)
X=[]
X = np.dstack((x1, x2))
XX = []
XX = np.hstack((np.reshape(x1[0, :], (3, 1)), np.reshape(x2[0, :], (3, 1))))
for i in range(1, 100):
    XX = np.dstack((XX, np.hstack((np.reshape(x1[i, :], (3, 1)), np.reshape(x2[i, :], (3, 1))))))

#dPCA
dpca = dPCA(labels='sn', n_components=1)
dpca.fit(XX)
PCA(copy=True, n_components=3, whiten=False)
print(pca.explained_variance_ratio_)

# Ploting the activations for the two stimuli within the original axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x1[:, 0], x1[:, 1], x1[:, 2], c='y')
ax.scatter(x2[:, 0], x2[:, 1], x2[:, 2], c='g')
ax.set_xlabel('Neuron1')
ax.set_ylabel('Neuron2')
ax.set_zlabel('Neuron3')

plt.show()






    