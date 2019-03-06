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
N, T, S = 3, 10, 2

n_trials = 100  

# Definition of the correlation matrix between the 3 neurons and means
a = np.random.randint(5, size=(N, N))  # random matrix
np.fill_diagonal(a, 1)
corr = np.tril(a) + np.tril(a, -1).T

mean1 = [0, 0, 0]
mean2 = [5, 3, 2]
means = [mean1, mean2]

# Getting the trial by trial data for each time point
# from a multivariate distribution with mean = means and covariance = corr
Xtrial = np.zeros((n_trials, N, T, S))
for t in range(T):
    x1 = np.random.multivariate_normal(means[0], corr, size=n_trials)
    x2 = np.random.multivariate_normal(means[1], corr, size=n_trials)
    Xtrial[:, :, t, 0] = x1
    Xtrial[:, :, t, 1] = x2


# trial-average data
X = np.mean(Xtrial, 0)
# centered data
Xc = X-np.mean(X.reshape((N, -1)), 1)[:, None, None]

#XX = []
#XX = np.hstack((np.reshape(x1[0, :], (3, 1)), np.reshape(x2[0, :], (3, 1))))
#for i in range(1, 100):
#    XX = np.dstack((XX, np.hstack((np.reshape(x1[i, :], (3, 1)), np.reshape(x2[i, :], (3, 1))))))


# Ploting the activations for the two stimuli within the original axis for each trial
# and the average across trials with a black x
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#for t in range(T):
#    ax.scatter(Xc[0, t, 0], Xc[1, t, 0], Xc[2, t, 0], c='b', s=2*t+2)
#    ax.scatter(Xc[0, t, 1], Xc[1, t, 1], Xc[2, t, 1], c='r', s=2*t+2) 
#ax.set_xlabel('Neuron1')
#ax.set_ylabel('Neuron2')
#ax.set_zlabel('Neuron3')
#
#plt.show()
#


#dPCA
dpca = dPCA(labels='st', regularizer='auto')
dpca.protect = ['t']

# Fit and transform the data to the new axes
X_proj = dpca.fit_transform(Xc, Xtrial)

# ploting the new projection agains the old 
fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')
for t in range(T):
    ax1.scatter(Xc[0, t, 0], Xc[1, t, 0], Xc[2, t, 0], c='g', s=2*t+5)
    ax1.scatter(Xc[0, t, 1], Xc[1, t, 1], Xc[2, t, 1], c='y', s=2*t+5) 
    ax1.scatter(X_proj['t'][0, t, 0], X_proj['t'][1, t, 0], X_proj['t'][2, t, 0], 
                c='g', s=2*t+5, marker='x')
    ax1.scatter(X_proj['t'][0, t, 1], X_proj['t'][1, t, 1], X_proj['t'][2, t, 1], 
                c='y', s=2*t+5, marker='x')
    
    ax1.scatter(X_proj['st'][0, t, 0], X_proj['st'][1, t, 0], X_proj['st'][2, t, 0], 
                c='g', s=2*t+5, marker='v')
    ax1.scatter(X_proj['st'][0, t, 1], X_proj['st'][1, t, 1], X_proj['st'][2, t, 1], 
                c='y', s=2*t+5, marker='v')
    
    ax1.scatter(X_proj['s'][0, t, 0], X_proj['s'][1, t, 0], X_proj['s'][2, t, 0], 
                c='g', s=2*t+5, marker='+')
    ax1.scatter(X_proj['s'][0, t, 1], X_proj['s'][1, t, 1], X_proj['s'][2, t, 1], 
                c='y', s=2*t+5, marker='+')
ax1.set_xlabel('Neuron1')
ax1.set_ylabel('Neuron2')
ax1.set_zlabel('Neuron3')

plt.show()





    