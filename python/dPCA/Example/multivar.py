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

# number of neurons and stimuli
N, S = 3, 2

# Definition of the correlation matrix between the 3 neurons and means
a = np.random.randint(5, size=(N, N))  # random matrix
np.fill_diagonal(a, 1)
corr = np.tril(a) + np.tril(a, -1).T

mean1 = [0, 0, 0]
mean2 = [10, 10, 10]
means = [mean1, mean2]

# Getting the trial by trial data from a multivariate distribution with 
# mean = means and covariance = corr
n = 100
x1 = np.random.multivariate_normal(means[0], corr, size=n)
x2 = np.random.multivariate_normal(means[1], corr, size=n)

# X matrix has dimentions (n trials, 3 neurons, 2 stimuli)
# i.e. activations for each trial of each neurons for each stimulus
Xtrial = []
Xtrial = np.dstack((x1, x2))

# trial-average data
X = np.mean(Xtrial, 0)
# centered data
Xc = X-np.mean(X.reshape((N, -1)), 1)[:, None]

#XX = []
#XX = np.hstack((np.reshape(x1[0, :], (3, 1)), np.reshape(x2[0, :], (3, 1))))
#for i in range(1, 100):
#    XX = np.dstack((XX, np.hstack((np.reshape(x1[i, :], (3, 1)), np.reshape(x2[i, :], (3, 1))))))


# Ploting the activations for the two stimuli within the original axis for each trial
# and the average across trials with a black x
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x1[:, 0], x1[:, 1], x1[:, 2], c='y')
ax.scatter(x2[:, 0], x2[:, 1], x2[:, 2], c='g')
ax.scatter(X[0, 0], X[1, 0], X[2, 0], c='k', s=40, marker='x', lw=7)
ax.scatter(X[0, 1], X[1, 1], X[2, 1], c='k', s=40, marker='x', lw=7)
ax.scatter(Xc[0, 0], Xc[1, 0], Xc[2, 0], c='b', s=40, marker='x', lw=7)
ax.scatter(Xc[0, 1], Xc[1, 1], Xc[2, 1], c='b', s=40, marker='x', lw=7)
ax.set_xlabel('Neuron1')
ax.set_ylabel('Neuron2')
ax.set_zlabel('Neuron3')

plt.show()



#dPCA
dpca = dPCA(labels='s', regularizer='auto')
dpca.protect = ['s']

# Fit and transform the data to the new axes
X_proj = dpca.fit_transform(Xc, Xtrial)

# ploting the new projection agains the old 
fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')
ax1.scatter(Xc[0, 0], Xc[1, 0], Xc[2, 0], c='b', s=40, marker='x', lw=7)
ax1.scatter(Xc[0, 1], Xc[1, 1], Xc[2, 1], c='b', s=40, marker='x', lw=7)
ax1.scatter(X_proj['s'][0, 0], X_proj['s'][1, 0], X_proj['s'][2, 0], c='r', s=40, marker='x', lw=7)
ax1.scatter(X_proj['s'][0, 1], X_proj['s'][1, 1], X_proj['s'][2, 1], c='r', s=40, marker='x', lw=7)
ax1.set_xlabel('Neuron1')
ax1.set_ylabel('Neuron2')
ax1.set_zlabel('Neuron3')

plt.show()





    