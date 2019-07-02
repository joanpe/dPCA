#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 14:44:51 2019

@author: joan
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import pdb
import sys
import os
# from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
PATH_SAVE = '/home/joan/dPCA/python/dPCA/dualtask/Figures_noise_0.0/'
#PATH_LOAD = '/home/joan/dPCA/python/dPCA/dualtask/Figures_noise_0.0/'
PATH_LOAD = '/home/joan/cluster_home/dPCA/python/dPCA/dualtask/'
sys.path.insert(0, PATH_LOAD)
from sklearn.decomposition import PCA
from scipy.stats import ttest_ind
import seaborn as sns


def princomp(data,n_components=None):
	pca=PCA()
	pca.fit(data)
	latent = pca.explained_variance_
	coeff = pca.components_
	score = pca.transform(data)

	return (coeff,score,latent)


# Parameters of the RNN
# Noise range for the input to the RNN
noise_rng = np.array([0.0])
noise = noise_rng[0]
#noise_rng = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
# Time of appearence of the go- no go task. 0 for no task. if gng_rng = [-1] 
# then it runs a ramdom trial either of the dualtask, dpa alone or gng alone.
gng_rng = np.array(-1)
gng = gng_rng
#gng_rng = np.array([0, 10])
lamb = np.array([0.0])
l = lamb[0]
#lamb = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1])
delay_max = np.array([0])
delay = delay_max[0]
num_neurons = np.array([64])
neuron = num_neurons[0]
load_dir = os.path.join(PATH_LOAD, 'data_trainedwithnoise')

INST = [4,21,24,35,42] # number of RNN instances
n_plot = 36 #Number of example plots to show
n_time = 20 # trial duration
n_states = 64 # number of hidden neurons
gng_time = 10 # time at which the distractor appears
n_batch = 2048
var0perf = []
var0 = []

#for inst in INST:
#    # Get the data from the 18th instance RNN
#    save_dir = os.path.join(PATH_SAVE, 'PCA')
#    savesave_dir = os.path.join(save_dir, 'inst'+str(inst))
#    
#    # Load data gathered from the RNN
#    data = np.load(os.path.join(PATH_LOAD,'data' + str(inst)+ '.npz'))
#    
#    
#    state = data['state'] # activations of the RNN for 2048trials, 20time steps per trial and 64 neurons
#    # get the accuracies
#    acc_dpa_trial = np.array(data['acc'][8][inst])*1 # accuracy of the dpa task for each trial
#    acc_dual_trial = np.array(data['acc'][7][inst])*1 # accuracy of the dpa task for each trial
#    task = data['task']
#    acc_dpa = data['acc'][5][inst]
#    acc_dual = data['acc'][3][inst]
#    stim_conf = np.array(data['stim_conf'])
#    
#    if np.logical_and(acc_dpa > 0.55, acc_dual > 0.55):
#        '''ANALYSES 2: PCA'''
#        # Activations of neurons for time = dpa2time -1 :
#        state_mean = state[:, 14, :]
#        
#        
#        # Compute the PCA
#        [coeff,score,latent] = princomp(state_mean)
#        #PCA analyses. coeff=linear combination
#        #. score=representation of the state_mean in the different 
#        # componets. latent=eigenvalues= the amount of variance explained for each 
#        # component
#        l= latent[:10] #first 10 components
#        #save var explained for acc<1 or acc=1:
#        if np.logical_or(acc_dpa<1, acc_dual<1):
#            var0.append(l[0]/sum (l))
#        
#        else:
#            var0perf.append(l[0]/sum(l))        
#        
#        fig = plt.figure()
#        plt.bar (range(10), l/sum (l)) # bar plot of the normalization of the amount
#        # of variance
#        plt.xticks(range(10))
#        plt.xlabel('PCs')
#        plt.ylabel('Amount of variance explained')
#        #plt.legend()
#        plt.gcf()
#        
#        if os.path.isdir(savesave_dir) is False:
#            os.mkdir(savesave_dir)
#            fig.savefig(os.path.join(savesave_dir, 'PCAvariance.png'))
#        else:
#            fig.savefig(os.path.join(savesave_dir, 'PCAvariance.png'))    
#        
#        plt.close()
#        
#        #Plot of the 2 firsts PCA components
#        fig = plt.figure()
#        plt.scatter (score[:,0],score[:,1]) #plot the firsts 2 components 
#        plt.xlabel('1st PC')
#        plt.ylabel('2nd PC')
#        plt.gcf()
#        
#        if os.path.isdir(savesave_dir) is False:
#            os.mkdir(savesave_dir)
#            fig.savefig(os.path.join(savesave_dir, 'PCAscores2d.png'))
#        else:
#            fig.savefig(os.path.join(savesave_dir, 'PCAscores2d.png'))  
#        
#        plt.close()
#        
#        index_dual = np.where(task==0)[0]
#        index_dpa = np.where(task==1)[0]
#        index_gng = np.where(task==2)[0]
#         
#        #Plot of the 2 firsts PCA components with colors for different conditions
#        fig = plt.figure()
#        plt.scatter(score[index_dual, 0],score[index_dual, 1], c='r') #plot the firsts 2 components 
#        plt.scatter(score[index_dpa, 0],score[index_dpa, 1], c='g')
#        plt.scatter(score[index_gng, 0],score[index_gng, 1], c='b')
#        plt.xlabel('1st PC')
#        plt.ylabel('2nd PC')
#        plt.gcf()
#        
#        if os.path.isdir(save_dir) is False:
#            os.mkdir(savesave_dir)
#            fig.savefig(os.path.join(savesave_dir, 'PCAscores2dcolor.png'))
#        else:
#            fig.savefig(os.path.join(savesave_dir, 'PCAscores2dcolor.png'))  
#        
#        plt.close()
#        
#        # ploting score but with different colors for the different type of trials. 
#        #Score have each representation to the new pc for each activation 
#        # of each time series in rech
#        # I add some noise in order to see all 2048 data point not overlapped
#        noisedual_x = np.random.normal(0, 0.003, np.shape(index_dual)[0])
#        noisedual_y = np.random.normal(0, 0.003, np.shape(index_dual)[0])
#        noisedual_z = np.random.normal(0, 0.003, np.shape(index_dual)[0])
#        noisedpa_x = np.random.normal(0, 0.003, np.shape(index_dpa)[0])
#        noisedpa_y = np.random.normal(0, 0.003, np.shape(index_dpa)[0])
#        noisedpa_z = np.random.normal(0, 0.003, np.shape(index_dpa)[0])
#        noisegng_x = np.random.normal(0, 0.003, np.shape(index_gng)[0])
#        noisegng_y = np.random.normal(0, 0.003, np.shape(index_gng)[0])
#        noisegng_z = np.random.normal(0, 0.003, np.shape(index_gng)[0])
#        
#        fig = plt.figure()
#        ax = fig.add_subplot(111, projection='3d')
#        ax.scatter (score[index_dual, 0]+noisedual_x, score[index_dual, 1]+noisedual_y,
#                    score[index_dual, 2]+noisedual_z, label='dual-task trials',
#                    color='r', s=1)
#        ax.scatter (score[index_dpa, 0]+noisedpa_x, score[index_dpa, 1]+noisedpa_y,
#                    score[index_dpa, 2]+noisedpa_z, label='DPA task trials',
#                    color='g', s=1)
#        ax.scatter (score[index_gng, 0]+noisegng_x, score[index_gng, 1]+noisegng_y,
#                    score[index_gng, 2]+noisegng_z, label='go/no-go task triasl',
#                    color='b', s=1)
#        
#        #ax.scatter (score[index_dual, 0], score[index_dual, 1],
#        #            score[index_dual, 2], label='dual-task trials',
#        #            color='r', s=1)
#        #ax.scatter (score[index_dpa, 0], score[index_dpa, 1],
#        #            score[index_dpa, 2], label='DPA task trials',
#        #            color='g', s=1)
#        #ax.scatter (score[index_gng, 0], score[index_gng, 1],
#        #            score[index_gng, 2], label='go/no-go task triasl',
#        #            color='b', s=1)
#        
#        ax.set_xlabel('1st PC')
#        ax.set_ylabel('2nd PC')
#        ax.set_zlabel('3rd PC')
#        plt.legend()
#        plt.gcf()
#        
#        if os.path.isdir(savesave_dir) is False:
#            os.mkdir(savesave_dir)
#            fig.savefig(os.path.join(savesave_dir, 'PCAscores3d.png'))
#        else:
#            fig.savefig(os.path.join(savesave_dir, 'PCAscores3d.png')) 
#        
#        plt.close()
#
#mean_var0 = np.mean(var0)
#std = np.std(var0)
#mean_var0perf = np.mean(var0perf)
#stdperf = np.std(var0perf)
#
#fig = plt.figure()
#ax = fig.add_subplot(111)
#ax.errorbar([0,1], [mean_var0, mean_var0perf], yerr=[std, stdperf], marker='o')
##misses look for errorbars style
#ax.set_xticks([0,1])
#ax.set_xticklabels(['< 1', '= 1'])
#ax.set_xlabel('accuracy dual')
#ax.set_ylabel('variance of 1rs PC')
#fig.savefig(os.path.join(save_dir, 'Var0vsAcc.png'))
#
#plt.show()

for inst in range(INST):
    # Get the data from the 18th instance RNN
    save_dir = os.path.join(PATH_SAVE, 'PCA')
    savesave_dir = os.path.join(save_dir, 'inst'+str(inst))
    
    # Load data gathered from the RNN
    data = np.load(os.path.join(load_dir, 'data_' + str(gng) + '_'
                                  + str(l) + '_' + str(delay)
                                  + '_i' + str(INST) + '_n' + str(noise_rng[0])
                                  + '-' + str(noise_rng[-1])
                                  + '_neu' + str(num_neurons[0])
                                  + '-' + str(num_neurons[-1]) + '.npz'))
    
    
    state = data['state'][inst] # activations of the RNN for 2048trials, 20time steps per trial and 64 neurons
    # get the accuracies
    acc_dpa_trial = np.array(data['acc'][0][8][inst])*1 # accuracy of the dpa task for each trial
    acc_dual_trial = np.array(data['acc'][0][7][inst])*1 # accuracy of the dpa task for each trial
    task = data['task'][inst]
    acc_dpa = data['acc'][0][5][inst]
    acc_dual = data['acc'][0][3][inst]
    stim_conf = np.array(data['stim_conf'][inst])
    
    if np.logical_and(acc_dpa > 0.55, acc_dual > 0.55):
        '''ANALYSES 2: PCA'''
        # Activations of neurons for time = dpa2time -1 :
        state_mean = state[:, 14, :]
        
        # Compute the PCA
        [coeff,score,latent] = princomp(state_mean)
        #PCA analyses. coeff=linear combination
        #. score=representation of the state_mean in the different 
        # componets. latent=eigenvalues= the amount of variance explained for each 
        # component
        l= latent[:10] #first 10 components
        #save var explained for acc<1 or acc=1:
        if np.logical_or(acc_dpa<1, acc_dual<1):
            var0.append(l[0]/sum (l))
        
        else:
            var0perf.append(l[0]/sum(l))        
        
        fig = plt.figure()
        plt.bar (range(10), l/sum (l)) # bar plot of the normalization of the amount
        # of variance
        plt.xticks(range(10))
        plt.xlabel('PCs')
        plt.ylabel('Amount of variance explained')
        #plt.legend()
        plt.gcf()
        
        if os.path.isdir(savesave_dir) is False:
            os.mkdir(savesave_dir)
            fig.savefig(os.path.join(savesave_dir, 'PCAvariance.png'))
        else:
            fig.savefig(os.path.join(savesave_dir, 'PCAvariance.png'))    
        
        plt.close()
        
        #Plot of the 2 firsts PCA components
        fig = plt.figure()
        plt.scatter (score[:,0],score[:,1]) #plot the firsts 2 components 
        plt.xlabel('1st PC')
        plt.ylabel('2nd PC')
        plt.gcf()
        
        if os.path.isdir(savesave_dir) is False:
            os.mkdir(savesave_dir)
            fig.savefig(os.path.join(savesave_dir, 'PCAscores2d.png'))
        else:
            fig.savefig(os.path.join(savesave_dir, 'PCAscores2d.png'))  
        
        plt.close()
        
        index_dual = np.where(task==0)[0]
        index_dpa = np.where(task==1)[0]
        index_gng = np.where(task==2)[0]
         
        #Plot of the 2 firsts PCA components with colors for different conditions
        fig = plt.figure()
        plt.scatter(score[index_dual, 0],score[index_dual, 1], c='r') #plot the firsts 2 components 
        plt.scatter(score[index_dpa, 0],score[index_dpa, 1], c='g')
        plt.scatter(score[index_gng, 0],score[index_gng, 1], c='b')
        plt.xlabel('1st PC')
        plt.ylabel('2nd PC')
        plt.gcf()
        
        if os.path.isdir(save_dir) is False:
            os.mkdir(savesave_dir)
            fig.savefig(os.path.join(savesave_dir, 'PCAscores2dcolor.png'))
        else:
            fig.savefig(os.path.join(savesave_dir, 'PCAscores2dcolor.png'))  
        
        plt.close()
        
        # ploting score but with different colors for the different type of trials. 
        #Score have each representation to the new pc for each activation 
        # of each time series in rech
        # I add some noise in order to see all 2048 data point not overlapped
        noisedual_x = np.random.normal(0, 0.003, np.shape(index_dual)[0])
        noisedual_y = np.random.normal(0, 0.003, np.shape(index_dual)[0])
        noisedual_z = np.random.normal(0, 0.003, np.shape(index_dual)[0])
        noisedpa_x = np.random.normal(0, 0.003, np.shape(index_dpa)[0])
        noisedpa_y = np.random.normal(0, 0.003, np.shape(index_dpa)[0])
        noisedpa_z = np.random.normal(0, 0.003, np.shape(index_dpa)[0])
        noisegng_x = np.random.normal(0, 0.003, np.shape(index_gng)[0])
        noisegng_y = np.random.normal(0, 0.003, np.shape(index_gng)[0])
        noisegng_z = np.random.normal(0, 0.003, np.shape(index_gng)[0])
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter (score[index_dual, 0]+noisedual_x, score[index_dual, 1]+noisedual_y,
                    score[index_dual, 2]+noisedual_z, label='dual-task trials',
                    color='r', s=1)
        ax.scatter (score[index_dpa, 0]+noisedpa_x, score[index_dpa, 1]+noisedpa_y,
                    score[index_dpa, 2]+noisedpa_z, label='DPA task trials',
                    color='g', s=1)
        ax.scatter (score[index_gng, 0]+noisegng_x, score[index_gng, 1]+noisegng_y,
                    score[index_gng, 2]+noisegng_z, label='go/no-go task triasl',
                    color='b', s=1)
        
        #ax.scatter (score[index_dual, 0], score[index_dual, 1],
        #            score[index_dual, 2], label='dual-task trials',
        #            color='r', s=1)
        #ax.scatter (score[index_dpa, 0], score[index_dpa, 1],
        #            score[index_dpa, 2], label='DPA task trials',
        #            color='g', s=1)
        #ax.scatter (score[index_gng, 0], score[index_gng, 1],
        #            score[index_gng, 2], label='go/no-go task triasl',
        #            color='b', s=1)
        
        ax.set_xlabel('1st PC')
        ax.set_ylabel('2nd PC')
        ax.set_zlabel('3rd PC')
        plt.legend()
        plt.gcf()
        
        if os.path.isdir(savesave_dir) is False:
            os.mkdir(savesave_dir)
            fig.savefig(os.path.join(savesave_dir, 'PCAscores3d.png'))
        else:
            fig.savefig(os.path.join(savesave_dir, 'PCAscores3d.png')) 
        
        plt.close()
        

mean_var0 = np.mean(var0)
std = np.std(var0)
mean_var0perf = np.mean(var0perf)
stdperf = np.std(var0perf)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.errorbar([0,1], [mean_var0, mean_var0perf], yerr=[std, stdperf], marker='o')
#misses look for errorbars style
ax.set_xticks([0,1])
ax.set_xticklabels(['< 1', '= 1'])
ax.set_xlabel('accuracy dual')
ax.set_ylabel('variance of 1rs PC')
fig.savefig(os.path.join(save_dir, 'Var0vsAcc.png'))

plt.show()

        
        
#
#
## Mean activations of neurons across time:
#state_mean = state.mean(axis=1)
#
##Want to see if there are clusters of neurons depending on the 3 task type within my trials
#fig = plt.figure()
#plt.plot(state_mean[0, :])
#plt.plot(state_mean[2, :])
#plt.plot(state_mean[100, :])
#plt.plot(state_mean[200, :])
#
## Compute the PCA
#[coeff,score,latent] = princomp(state_mean)
##PCA analyses. coeff=linear combination
##. score=representation of the state_mean in the different 
## componets. latent=eigenvalues= the amount of variance explained for each 
## component
#fig = plt.figure()
#l= latent[:10] #first 10 components
#plt.bar (range(10), l/sum (l)) # bar plot of the normalization of the amount
## of variance
#plt.xticks(range(10))
#plt.xlabel('PCs')
#plt.ylabel('Amount of variance explained')
##plt.legend()
#plt.gcf()
#plt.show()
#
#if os.path.isdir(savesave_dir) is False:
#    os.mkdir(savesave_dir)
#    fig.savefig(os.path.join(savesave_dir, 'PCAvariance.png'))
#else:
#    fig.savefig(os.path.join(savesave_dir, 'PCAvariance.png'))    
#
##Plot of the 2 firsts PCA components
#fig = plt.figure()
#plt.scatter (score[:,0],score[:,1]) #plot the firsts 2 components 
#plt.xlabel('1st PC')
#plt.ylabel('2nd PC')
#plt.gcf()
#plt.show()
#
#if os.path.isdir(savesave_dir) is False:
#    os.mkdir(savesave_dir)
#    fig.savefig(os.path.join(savesave_dir, 'PCAscores2d.png'))
#else:
#    fig.savefig(os.path.join(savesave_dir, 'PCAscores2d.png'))  
#
#
#
#index_dual = np.where(task==0)[0]
#index_dpa = np.where(task==1)[0]
#index_gng = np.where(task==2)[0]
# 
##Plot of the 2 firsts PCA components with colors for different conditions
#fig = plt.figure()
#plt.scatter(score[index_dual, 0],score[index_dual, 1], c='r') #plot the firsts 2 components 
#plt.scatter(score[index_dpa, 0],score[index_dpa, 1], c='g')
#plt.scatter(score[index_gng, 0],score[index_gng, 1], c='b')
#plt.xlabel('1st PC')
#plt.ylabel('2nd PC')
#plt.gcf()
#plt.show()
#
#if os.path.isdir(save_dir) is False:
#    os.mkdir(savesave_dir)
#    fig.savefig(os.path.join(savesave_dir, 'PCAscores2dcolor.png'))
#else:
#    fig.savefig(os.path.join(savesave_dir, 'PCAscores2dcolor.png'))  
#
#
## ploting score but with different colors for the different type of trials. 
##Score have each representation to the new pc for each activation 
## of each time series in rech
## I add some noise in order to see all 2048 data point not overlapped
#noisedual_x = np.random.normal(0, 0.003, np.shape(index_dual)[0])
#noisedual_y = np.random.normal(0, 0.003, np.shape(index_dual)[0])
#noisedual_z = np.random.normal(0, 0.003, np.shape(index_dual)[0])
#noisedpa_x = np.random.normal(0, 0.003, np.shape(index_dpa)[0])
#noisedpa_y = np.random.normal(0, 0.003, np.shape(index_dpa)[0])
#noisedpa_z = np.random.normal(0, 0.003, np.shape(index_dpa)[0])
#noisegng_x = np.random.normal(0, 0.003, np.shape(index_gng)[0])
#noisegng_y = np.random.normal(0, 0.003, np.shape(index_gng)[0])
#noisegng_z = np.random.normal(0, 0.003, np.shape(index_gng)[0])
#
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.scatter (score[index_dual, 0]+noisedual_x, score[index_dual, 1]+noisedual_y,
#            score[index_dual, 2]+noisedual_z, label='dual-task trials',
#            color='r', s=1)
#ax.scatter (score[index_dpa, 0]+noisedpa_x, score[index_dpa, 1]+noisedpa_y,
#            score[index_dpa, 2]+noisedpa_z, label='DPA task trials',
#            color='g', s=1)
#ax.scatter (score[index_gng, 0]+noisegng_x, score[index_gng, 1]+noisegng_y,
#            score[index_gng, 2]+noisegng_z, label='go/no-go task triasl',
#            color='b', s=1)
#
##ax.scatter (score[index_dual, 0], score[index_dual, 1],
##            score[index_dual, 2], label='dual-task trials',
##            color='r', s=1)
##ax.scatter (score[index_dpa, 0], score[index_dpa, 1],
##            score[index_dpa, 2], label='DPA task trials',
##            color='g', s=1)
##ax.scatter (score[index_gng, 0], score[index_gng, 1],
##            score[index_gng, 2], label='go/no-go task triasl',
##            color='b', s=1)
#
#ax.set_xlabel('1st PC')
#ax.set_ylabel('2nd PC')
#ax.set_zlabel('3rd PC')
#plt.legend()
#plt.gcf()
#plt.show()
#
#if os.path.isdir(savesave_dir) is False:
#    os.mkdir(savesave_dir)
#    fig.savefig(os.path.join(savesave_dir, 'PCAscores3d.png'))
#else:
#    fig.savefig(os.path.join(savesave_dir, 'PCAscores3d.png')) 
#
##Porjections of each time onto the new pc axes
#f = plt.figure()
#ax = f.add_subplot(111, projection='3d')
#for t in range(n_time):
#    statePC = np.matmul(state[:, t, :], coeff)
##    ax.plot(statePC[index_dual, 0], statePC[index_dual, 1],
##            statePC[index_dual, 2], c=(0,t/20.0,0,1)) #plot the firsts 2 components 
#    ax.plot(statePC[index_dpa, 0], statePC[index_dpa, 1],
#            statePC[index_dpa, 2], c=(0,t/20.0,1,1))
##    ax.plot(statePC[index_gng, 0], statePC[index_gng, 1],
##            statePC[index_gng, 2], c=(1,t/20.0,0,1))
#
#
#
#ax.set_xlabel('1st PC')
#ax.set_ylabel('2nd PC')
#ax.set_zlabel('3rd PC')
#plt.gcf()
#plt.show()

#
#'''ANALYSES 1: RESAMPLING METHODS, MEAN AND MEDIAN'''
## Compute the means of accuracy in the network and see if that 
## mean for each subject is normally distributed
#
#
#fig = plt.figure()
#plt.hist(acc_dual, bins=40, alpha=0.5, color='red', label='dual task')
#plt.hist(acc_dpa, bins=40, alpha=0.5, color='green', label='dpa task')
#plt.plot(np.mean(acc_dual), -1, color='r', marker='v', label='Mean')
#plt.plot([np.mean(acc_dual)-np.std(acc_dual), np.mean(acc_dual)+np.std(acc_dual)],
#          [-1,-1], color='r')
#plt.plot(np.mean(acc_dpa), -2, color='g', marker='v')
#plt.plot([np.mean(acc_dpa)-np.std(acc_dpa), np.mean(acc_dpa)+np.std(acc_dpa)],
#          [-2,-2], color='g')
#
#md12 = np.median(acc_dpa)
#md34 = np.median(acc_dual)
#q25_12 = np.percentile(acc_dpa, 25)
#q75_12 = np.percentile(acc_dpa, 75)
#q25_34 = np.percentile(acc_dual, 25)
#q75_34 = np.percentile(acc_dual, 75)
#plt.plot(md12, 11, color='g', marker='*')
#plt.plot(md34, 12, color='r', marker='*', label='Median')
#plt.plot([q25_12, q75_12], [11,11], color='g')
#plt.plot([q25_34, q75_34], [12,12], color='r')
#plt.legend(loc='upper left')
#plt.xlabel('Accuracy')
#plt.ylabel('Number of subjects')
#plt.show()
#
#if os.path.isdir(save_dir) is False:
#    os.mkdir(save_dir)
#    fig.savefig(os.path.join(save_dir, 'hist_acc_mean_median.png'))
#else:
#    fig.savefig(os.path.join(save_dir, 'hist_acc_mean_median.png')) 
#
#
##Convert to gaussian
#acc_dualg = 10*np.log10(acc_dual)
#acc_dpag = 10*np.log10(acc_dpa)
#
#fig = plt.figure()
#plt.hist(acc_dualg, bins=40, alpha=0.5, color='red', label='dual task')
#plt.hist(acc_dpag, bins=40, alpha=0.5, color='green', label='dpa task')
#plt.plot(np.mean(acc_dualg), -1, color='r', marker='v', label='Mean')
#plt.plot([np.mean(acc_dualg)-np.std(acc_dualg), np.mean(acc_dualg)+np.std(acc_dualg)],
#          [-1,-1], color='r')
#plt.plot(np.mean(acc_dpag), -2, color='g', marker='v')
#plt.plot([np.mean(acc_dpag)-np.std(acc_dpag), np.mean(acc_dpag)+np.std(acc_dpag)],
#          [-2,-2], color='g')
#
#md12 = np.median(acc_dpag)
#md34 = np.median(acc_dualg)
#q25_12 = np.percentile(acc_dpag, 25)
#q75_12 = np.percentile(acc_dpag, 75)
#q25_34 = np.percentile(acc_dualg, 25)
#q75_34 = np.percentile(acc_dualg, 75)
#plt.plot(md12, 11, color='g', marker='*')
#plt.plot(md34, 12, color='r', marker='*', label='Median')
#plt.plot([q25_12, q75_12], [11,11], color='g')
#plt.plot([q25_34, q75_34], [12,12], color='r')
#plt.xlabel('10log10 Accuracy')
#plt.ylabel('Number of subjects')
#plt.legend()
#plt.show()
#
#if os.path.isdir(save_dir) is False:
#    os.mkdir(save_dir)
#    fig.savefig(os.path.join(save_dir, 'hist_acc_mean_mediangaussian.png'))
#else:
#    fig.savefig(os.path.join(save_dir, 'hist_acc_mean_mediangaussian.png')) 
#
##standard error of the mean
#sem = [np.std(acc_dualg)/np.sqrt(len(acc_dualg)), np.std(acc_dpag)/np.sqrt(len(acc_dpag))]
#
#
##printing the confidence interval for the gaussian distributed data
#print('Parametric')
#print ('mean dual task accuracy', np.mean(acc_dualg))
#print('confidence interval', np.mean(acc_dualg)-2*sem[0], np.mean(acc_dualg)+2*sem[0])
#print ('mean dpa task accuracy', np.mean(acc_dpag))
#print('confidence interval', np.mean(acc_dpag)-2*sem[1], np.mean(acc_dpag)+2*sem[1])
#
#
##Let's do it for no parametric statistics with bootstrap
##definig variables
#nn12 = acc_dual
#nn34 = acc_dpa
#l12 = len(nn12)
#l34 = len(nn34)
#md12 = []
#md34 = []
#
##for loop to make a vector of all the medians with the new reshuffled data
##randint reshuffles the indexes and then i compute the median for every 
##new index and append it in a vector
#for i in range(1000):
#  ind = np.random.randint(0, l12, l12)
#  md12.append(np.median(nn12[ind]))
#  ind2 = np.random.randint(0, l34, l34)
#  md34.append(np.median(nn34[ind2]))
#
##standart deviation and confidence interval for the vector of medians
#semn = [np.std(md12), np.std(md34)]
#ci12 = [np.percentile(md12, 2.5), np.percentile(md12, 97.5)]
#ci34 = [np.percentile(md34, 2.5), np.percentile(md34, 97.5)]
#
#print ('Non parametric')
#print ('median dual task accuracy', np.median(nn12))
#print ('confidence interval', ci12)
#print ('median dpa task accuracy', np.median(nn34))
#print ('confidence interval', ci34)
#
#
