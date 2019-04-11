'''
run_flipflop.py
Written using Python 2.7.12
@ Matt Golub, October 2018.
Please direct correspondence to mgolub@stanford.edu.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import pdb
import sys
import os
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
PATH = '/home/joan/dPCA/python/dPCA'
sys.path.insert(0, PATH) 
from DualTask import DualTask
from dPCA import dPCA

# *****************************************************************************
# STEP 1: Train an RNN to solve the dual task *********************************
# *****************************************************************************
noise_rng = np.array([0.05, 0.1, 0.15])
data_mat = np.zeros((5, noise_rng.shape[0]))
for inst in range(5):
    for noise in noise_rng:
        # Hyperparameters for AdaptiveLearningRate
        alr_hps = {'initial_rate': 0.1}
        
        # Hyperparameters for FlipFlop
        # See FlipFlop.py for detailed descriptions.
        hps = {
            'rnn_type': 'vanilla',
            'n_hidden': 256,
            'min_loss': 1e-6,  # 1e-4
            'min_learning_rate': 1e-5,
            'max_n_epochs': 10000,
            'do_restart_run': True,
            'log_dir': './logs/',
            'data_hps': {
                'n_batch': 2048,
                'n_time': 20,
                'n_bits': 6,
                'noise': noise,
                'gng_time': 10,
                'lamb': 0,
                'delay_max': 0},
            'alr_hps': alr_hps
            }

        dt = DualTask(**hps)

        dt.train()

# Get example state trajectories from the network
# Visualize inputs, outputs, and RNN predictions from example trials
        example_trials = dt.generate_dualtask_trials()
# f = dt.plot_trials(example_trials)

        n_bits = dt.hps.data_hps['n_bits']
        n_batch = dt.hps.data_hps['n_batch']
        n_states = dt.hps.n_hidden
        n_time = dt.hps.data_hps['n_time']
        is_lstm = dt.hps.rnn_type == 'lstm'
        gng_time = dt.hps.data_hps['gng_time']
        lamb = dt.hps.data_hps['lamb']
        delay_max = dt.hps.data_hps['delay_max']

        example_predictions = dt.predict(example_trials,
                                         do_predict_full_LSTM_state=is_lstm)

        summary = dt._train_batch(example_trials)
        loss = summary['loss']

        data_mat[inst, np.where(noise_rng == noise)[0]] = loss

        data = {'inst': inst, 'noise': noise, 'loss': loss}

        fig_dir = os.path.join(PATH, 'data')
        try:
            os.mkdir(fig_dir)
        except OSError:
            np.savez(os.path.join(fig_dir, 'noise_' + str(noise) +
                                  '_inst_' + str(inst)), **data)
        else:
            np.savez(os.path.join(fig_dir, 'noise_' + str(noise) +
                                  '_inst_' + str(inst)), **data)

# f.canvas.draw()
# *****************************************************************************
# STEP 2: Collect the predicted output data, reduce dimensionality and
#           visualize  *******************************************************
# *****************************************************************************





#
#'''Reordering of the example predictions in order to input to dPCA.'''
#
#'''S1 and S2 as a condition'''
## number of elements for S1 and for S2
#n0 = np.shape(np.where(example_trials['stim_conf'][:, 0] == 0)[0])[0]
#n1 = np.shape(np.where(example_trials['stim_conf'][:, 0] == 1)[0])[0]
#
## Arrays of the elements corresponding to S1 and S2 of different sizes
#predictions0 = np.zeros([n0, n_states, n_time])
#predictions1 = np.zeros([n1, n_states, n_time])
#
#
#for ind_state in range(n_states):
#    predictions0[:, ind_state, :] = example_predictions['state'][
#            example_trials['stim_conf'][:, 0] == 0, :, ind_state]
#
#    predictions1[:, ind_state, :] = example_predictions['state'][
#            example_trials['stim_conf'][:, 0] == 1, :, ind_state]
#
#predictions = np.stack((predictions0, predictions1), axis=2)
#
## trial-average data
#pred_mean = np.mean(predictions, 0)
#
## center data
#pred_mean -= np.mean(pred_mean.reshape((n_states, -1)), 1)[:, None, None]
#
#'''dPCA transform'''
#
#dpca = dPCA(labels='st', regularizer='auto')
#dpca.protect = ['t']
#
#Z = dpca.fit_transform(pred_mean, predictions)
#
## significance analyses:
#
#masks_s1 = dpca.significance_analysis(pred_mean, predictions,
#                                      axis='t', n_shuffles=100,
#                                      n_splits=10, n_consecutive=10)
#
#
#'''Ploting the activations of 3 neurons for each stimulus'''
#FIG_WIDTH = 6  # inches
#FIG_HEIGHT = 6  # inches
#FONT_WEIGHT = 'bold'
#
#
#'''Ploting the amplitude of the 1rst and 2nd components of each parameter
#across time'''
#c1, c2, c3 = 0, 1, 2
#time = np.arange(n_time)
#fig = plt.figure(figsize=(16, 7))
#fig.suptitle('gng time ' + str(gng_time) + ' max delay ' + str(delay_max) +
#             ' parametrization ' + str(lamb) +
#             '\nS1 / S2 data projected onto dPCA decoder axis')
#
#plt.subplots_adjust(hspace=0.5)
#plt.subplot(331)
#for s in range(2):
#    plt.plot(time, Z['t'][c1, s, :])
#plt.title(str(c1+1)+'st time component')
#
#
#plt.subplot(334)
#for s in range(2):
#    plt.plot(time, Z['t'][c2, s, :])
#plt.title(str(c2+1)+'d time component')
#plt.xlabel('time')
#
#plt.subplot(337)
#for s in range(2):
#    plt.plot(time, Z['t'][c3, s, :])
#plt.title(str(c3+1)+'d time component')
#plt.xlabel('time')
#
#plt.subplot(332)
#for s in range(2):
#    plt.plot(time, Z['s'][c1, s, :])
#plt.title(str(c1+1)+'st stimulus component')
#
#plt.subplot(335)
#for s in range(2):
#    plt.plot(time, Z['s'][c2, s, :])
#plt.title(str(c2+1)+'nd stimulus component')
#plt.xlabel('time')
#
#plt.subplot(338)
#for s in range(2):
#    plt.plot(time, Z['s'][c3, s, :])
#plt.title(str(c3+1)+'nd stimulus component')
#plt.xlabel('time')
#
#plt.subplot(333)
#for s in range(2):
#    plt.plot(time, Z['st'][c1, s, :])
#plt.title(str(c1+1)+'st mixing component')
#
#plt.subplot(336)
#for s in range(2):
#    plt.plot(time, Z['st'][c2, s, :])
#plt.title(str(c2+1)+'nd mixing component')
#plt.xlabel('time')
#plt.show()
#
#plt.subplot(339)
#for s in range(2):
#    plt.plot(time, Z['st'][c3, s, :])
#plt.title(str(c3+1)+'nd mixing component')
#plt.xlabel('time')
#plt.show()
#
#fig_dir = os.path.join('/home/joan/Desktop/dPCAfig/', str(gng_time) + '_' +
#                       str(delay_max) + '_' + str(lamb) + '/')
#try:
#    os.mkdir(fig_dir)
#except OSError:
#    plt.savefig(os.path.join(fig_dir, 'S1S2time.png'))
#else:
#    plt.savefig(os.path.join(fig_dir, 'S1S2time.png'))
#
#'''Ploting the 2 firsts demixed components for time, stimulus, and mixing'''
#
#
#fig = plt.figure(figsize=(15, FIG_HEIGHT), tight_layout=True)
#fig.suptitle('gng time ' + str(gng_time) + ' max delay ' + str(delay_max) +
#             ' parametrization ' + str(lamb) +
#             '\nS1 / S2 dPCA components')
#
#plt.subplot(131)
#for s in range(2):
#    plt.scatter(Z['t'][c1, s, :], Z['t'][c2, s, :], s=2*time+1)
## plot(E['t'], c='k', lw=5)
#plt.xlabel(str(c1+1)+'st time component')
#plt.ylabel(str(c2+1)+'nd time component')
#
#plt.subplot(132)
#for s in range(2):
#    plt.scatter(Z['s'][c1, s, :], Z['s'][c2, s, :], s=2*time+1)
## plot(E['s'], c='k', lw=5)
#plt.xlabel(str(c1+1)+'st stimulus component')
#plt.ylabel(str(c2+1)+'nd stimulus component')
#
#plt.subplot(133)
#for s in range(2):
#    plt.scatter(Z['st'][c1, s, :], Z['st'][c2, s, :], s=2*time+1)
## plot(E['s'], c='k', lw=5)
#plt.xlabel(str(c1+1)+'st mixing component')
#plt.ylabel(str(c2+1)+'nd mixing component')
#plt.show()
#
#plt.savefig(os.path.join(fig_dir, 'S1S2comp.png'))
#
#
#
#'''S3 and S4 as a condition'''
## number of elements for S1 and for S2
#n0 = np.shape(np.where(example_trials['stim_conf'][:, 1] == 0)[0])[0]
#n1 = np.shape(np.where(example_trials['stim_conf'][:, 1] == 1)[0])[0]
#
## Arrays of the elements corresponding to S1 and S2 of different sizes
#predictions0 = np.zeros([n0, n_states, n_time])
#predictions1 = np.zeros([n1, n_states, n_time])
#
#
#for ind_state in range(n_states):
#    predictions0[:, ind_state, :] = example_predictions['state'][
#            example_trials['stim_conf'][:, 1] == 0, :, ind_state]
#
#    predictions1[:, ind_state, :] = example_predictions['state'][
#            example_trials['stim_conf'][:, 1] == 1, :, ind_state]
#
#predictions = np.stack((predictions0, predictions1), axis=2)
#
## trial-average data
#pred_mean = np.mean(predictions, 0)
#
## center data
#pred_mean -= np.mean(pred_mean.reshape((n_states, -1)), 1)[:, None, None]
#
#'''dPCA transform'''
#
#dpca = dPCA(labels='st', regularizer='auto')
#dpca.protect = ['t']
#
#Z = dpca.fit_transform(pred_mean, predictions)
#
## Significance analyses
#
#masks_s3 = dpca.significance_analysis(pred_mean, predictions,
#                                      axis='t', n_shuffles=100,
#                                      n_splits=10, n_consecutive=10)
#
#
#'''Ploting the activations of 3 neurons for each stimulus'''
#FIG_WIDTH = 6  # inches
#FIG_HEIGHT = 6  # inches
#FONT_WEIGHT = 'bold'
#
#
## fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT), tight_layout=True)
##
## ax = fig.add_subplot(111, projection='3d')
## cmap = mpl.cm.jet
## for s in range(2):
##    for t in range(n_time):
##        ax.scatter(pred_mean[0, t, s], pred_mean[1, t, s], pred_mean[2, t, s],
##                   s=t/3+1, color=cmap(s / float(2)))
## plt.show()
#
#'''Ploting the amplitude of the 1rst and 2nd components of each parameter
#across time'''
#c1, c2, c3 = 0, 1, 2
#time = np.arange(n_time)
#fig = plt.figure(figsize=(16, 7))
#fig.suptitle('gng time ' + str(gng_time) + ' max delay ' + str(delay_max) +
#             ' parametrization ' + str(lamb) +
#             '\nS3 / S4 data projected onto dPCA decoder axis')
#
#plt.subplots_adjust(hspace=0.5)
#plt.subplot(331)
#for s in range(2):
#    plt.plot(time, Z['t'][c1, s, :])
#plt.title(str(c1+1)+'st time component')
#
#
#plt.subplot(334)
#for s in range(2):
#    plt.plot(time, Z['t'][c2, s, :])
#plt.title(str(c2+1)+'d time component')
#plt.xlabel('time')
#
#plt.subplot(337)
#for s in range(2):
#    plt.plot(time, Z['t'][c3, s, :])
#plt.title(str(c3+1)+'d time component')
#plt.xlabel('time')
#
#plt.subplot(332)
#for s in range(2):
#    plt.plot(time, Z['s'][c1, s, :])
#plt.title(str(c1+1)+'st stimulus component')
#
#plt.subplot(335)
#for s in range(2):
#    plt.plot(time, Z['s'][c2, s, :])
#plt.title(str(c2+1)+'nd stimulus component')
#plt.xlabel('time')
#
#plt.subplot(338)
#for s in range(2):
#    plt.plot(time, Z['s'][c3, s, :])
#plt.title(str(c3+1)+'nd stimulus component')
#plt.xlabel('time')
#
#plt.subplot(333)
#for s in range(2):
#    plt.plot(time, Z['st'][c1, s, :])
#plt.title(str(c1+1)+'st mixing component')
#
#plt.subplot(336)
#for s in range(2):
#    plt.plot(time, Z['st'][c2, s, :])
#plt.title(str(c2+1)+'nd mixing component')
#plt.xlabel('time')
#plt.show()
#
#plt.subplot(339)
#for s in range(2):
#    plt.plot(time, Z['st'][c3, s, :])
#plt.title(str(c3+1)+'nd mixing component')
#plt.xlabel('time')
#plt.show()
#
#plt.savefig(os.path.join(fig_dir, 'S3S4time.png'))
#
#'''Ploting the 2 firsts demixed components for time, stimulus, and mixing'''
#
#
#fig = plt.figure(figsize=(15, FIG_HEIGHT), tight_layout=True)
#fig.suptitle('gng time ' + str(gng_time) + ' max delay ' + str(delay_max) +
#             ' parametrization ' + str(lamb) +
#             '\nS3 / S4 dPCA components')
#
#plt.subplot(131)
#for s in range(2):
#    plt.scatter(Z['t'][c1, s, :], Z['t'][c2, s, :], s=2*time+1)
## plot(E['t'], c='k', lw=5)
#plt.xlabel(str(c1+1)+'st time component')
#plt.ylabel(str(c2+1)+'nd time component')
#
#plt.subplot(132)
#for s in range(2):
#    plt.scatter(Z['s'][c1, s, :], Z['s'][c2, s, :], s=2*time+1)
## plot(E['s'], c='k', lw=5)
#plt.xlabel(str(c1+1)+'st stimulus component')
#plt.ylabel(str(c2+1)+'nd stimulus component')
#
#plt.subplot(133)
#for s in range(2):
#    plt.scatter(Z['st'][c1, s, :], Z['st'][c2, s, :], s=2*time+1)
## plot(E['s'], c='k', lw=5)
#plt.xlabel(str(c1+1)+'st mixing component')
#plt.ylabel(str(c2+1)+'nd mixing component')
#plt.show()
#
#
##
## for s in range(2):
##    plt.scatter(Z['s'][0, s], Z['s'][1, s])
###plt.xlabel('1st stimulus component')
###plt.ylabel('2nd stimulus component')
## plt.show()
#plt.savefig(os.path.join(fig_dir, 'S3S4comp.png'))
#
#'''S5 and S6 as a condition'''
## number of elements for S1 and for S2
#n0 = np.shape(np.where(example_trials['stim_conf'][:, 2] == 0)[0])[0]
#n1 = np.shape(np.where(example_trials['stim_conf'][:, 2] == 1)[0])[0]
#
## Arrays of the elements corresponding to S1 and S2 of different sizes
#predictions0 = np.zeros([n0, n_states, n_time])
#predictions1 = np.zeros([n1, n_states, n_time])
#
#
#for ind_state in range(n_states):
#    predictions0[:, ind_state, :] = example_predictions['state'][
#            example_trials['stim_conf'][:, 2] == 0, :, ind_state]
#
#    predictions1[:, ind_state, :] = example_predictions['state'][
#            example_trials['stim_conf'][:, 2] == 1, :, ind_state]
#
#predictions = np.stack((predictions0, predictions1), axis=2)
#
## trial-average data
#pred_mean = np.mean(predictions, 0)
#
## center data
#pred_mean -= np.mean(pred_mean.reshape((n_states, -1)), 1)[:, None, None]
#
#'''dPCA transform'''
#
#dpca = dPCA(labels='st', regularizer='auto')
#dpca.protect = ['t']
#
#Z = dpca.fit_transform(pred_mean, predictions)
#
## Significance analyses
#
#masks_s5, scores, sh_scores = dpca.significance_analysis(pred_mean,
#                                                         predictions,
#                                                         axis='t',
#                                                         n_shuffles=100,
#                                                         n_splits=10,
#                                                         n_consecutive=10,
#                                                         full=True)
#
#
#'''Ploting the activations of 3 neurons for each stimulus'''
#FIG_WIDTH = 6  # inches
#FIG_HEIGHT = 6  # inches
#FONT_WEIGHT = 'bold'     
#     
##fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT), tight_layout=True)
##
##ax = fig.add_subplot(111, projection='3d')
##cmap = mpl.cm.jet
##for s in range(2):
##    for t in range(n_time):
##        ax.scatter(pred_mean[0, t, s], pred_mean[1, t, s], pred_mean[2, t, s],
##                   s=t/3+1, color=cmap(s / float(2)))
##plt.show()
#
#'''Ploting the amplitude of the 1rst and 2nd components of each parameter
#across time'''
#c1, c2, c3 = 0, 1, 2
#time = np.arange(n_time)
#fig = plt.figure(figsize=(16, 7))
#fig.suptitle('gng time ' + str(gng_time) + ' max delay ' + str(delay_max) +
#             ' parametrization ' + str(lamb) +
#             '\nS5 / S6 data projected onto dPCA decoder axis')
#
#plt.subplots_adjust(hspace=0.5)
#plt.subplot(331)
#for s in range(2):
#    plt.plot(time, Z['t'][c1, s, :])
#plt.title(str(c1+1)+'st time component')
#
#
#plt.subplot(334)
#for s in range(2):
#    plt.plot(time, Z['t'][c2, s, :])
#plt.title(str(c2+1)+'d time component')
#plt.xlabel('time')
#
#plt.subplot(337)
#for s in range(2):
#    plt.plot(time, Z['t'][c3, s, :])
#plt.title(str(c3+1)+'d time component')
#plt.xlabel('time')
#
#plt.subplot(332)
#for s in range(2):
#    plt.plot(time, Z['s'][c1, s, :])
#plt.title(str(c1+1)+'st stimulus component')
#
#plt.subplot(335)
#for s in range(2):
#    plt.plot(time, Z['s'][c2, s, :])
#plt.title(str(c2+1)+'nd stimulus component')
#plt.xlabel('time')
#
#plt.subplot(338)
#for s in range(2):
#    plt.plot(time, Z['s'][c3, s, :])
#plt.title(str(c3+1)+'nd stimulus component')
#plt.xlabel('time')
#
#plt.subplot(333)
#for s in range(2):
#    plt.plot(time, Z['st'][c1, s, :])
#plt.title(str(c1+1)+'st mixing component')
#
#plt.subplot(336)
#for s in range(2):
#    plt.plot(time, Z['st'][c2, s, :])
#plt.title(str(c2+1)+'nd mixing component')
#plt.xlabel('time')
#plt.show()
#
#plt.subplot(339)
#for s in range(2):
#    plt.plot(time, Z['st'][c3, s, :])
#plt.title(str(c3+1)+'nd mixing component')
#plt.xlabel('time')
#plt.show()
#
#plt.savefig(os.path.join(fig_dir, 'S5S6time.png'))
#
#'''Ploting the 2 firsts demixed components for time, stimulus, and mixing'''
#
#
#fig = plt.figure(figsize=(15, FIG_HEIGHT), tight_layout=True)
#fig.suptitle('gng time ' + str(gng_time) + ' max delay ' + str(delay_max) +
#             ' parametrization ' + str(lamb) +
#             '\nS5 / S6 dPCA components')
#
#plt.subplot(131)
#for s in range(2):
#    plt.scatter(Z['t'][c1, s, :], Z['t'][c2, s, :], s=2*time+1)
## plot(E['t'], c='k', lw=5)
#plt.xlabel(str(c1+1)+'st time component')
#plt.ylabel(str(c2+1)+'nd time component')
#
#plt.subplot(132)
#for s in range(2):
#    plt.scatter(Z['s'][c1, s, :], Z['s'][c2, s, :], s=2*time+1)
## plot(E['s'], c='k', lw=5)
#plt.xlabel(str(c1+1)+'st stimulus component')
#plt.ylabel(str(c2+1)+'nd stimulus component')
#
#plt.subplot(133)
#for s in range(2):
#    plt.scatter(Z['st'][c1, s, :], Z['st'][c2, s, :], s=2*time+1)
## plot(E['s'], c='k', lw=5)
#plt.xlabel(str(c1+1)+'st mixing component')
#plt.ylabel(str(c2+1)+'nd mixing component')
#plt.show()
#
#
##for s in range(2):
##    plt.scatter(Z['s'][0, s], Z['s'][1, s])
###plt.xlabel('1st stimulus component')
###plt.ylabel('2nd stimulus component')
##plt.show()
#plt.savefig(os.path.join(fig_dir, 'S5S6comp.png'))
#
##'''Go and No Go as a condition'''
### number of elements for S1 and for S2
##n0 = np.shape(np.where(example_trials['stim_conf'][:, 3] == 0)[0])[0]
##n1 = np.shape(np.where(example_trials['stim_conf'][:, 3] == 1)[0])[0]
##
### Arrays of the elements corresponding to S1 and S2 of different sizes
##predictions0 = np.zeros([n0, n_states, n_time])
##predictions1 = np.zeros([n1, n_states, n_time])
##
##
##for ind_state in range(n_states):
##    predictions0[:, ind_state, :] = example_predictions['state'][
##            example_trials['stim_conf'][:, 3] == 0, :, ind_state]
##
##    predictions1[:, ind_state, :] = example_predictions['state'][
##            example_trials['stim_conf'][:, 3] == 1, :, ind_state]
##
##if n1 < n0:
##    predictions0 = np.delete(predictions0, predictions0[n1:n0-1], axis=0)
##else:
##    predictions1 = np.delete(predictions1, predictions1[n0:n1-1], axis=0)
##
##predictions = np.stack((predictions0, predictions1), axis=3)
##
### trial-average data
##pred_mean = np.mean(predictions, 0)
##
### center data
##pred_mean -= np.mean(pred_mean.reshape((n_states, -1)), 1)[:, None, None]
##
##'''dPCA transform'''
##
##dpca = dPCA(labels='ts', regularizer='auto')
##dpca.protect = ['s']
##
##Z = dpca.fit_transform(pred_mean, predictions)
##
##
##'''Ploting the activations of 3 neurons for each stimulus'''
##FIG_WIDTH = 6  # inches
##FIG_HEIGHT = 6  # inches
##FONT_WEIGHT = 'bold'
##        
##        
###fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT), tight_layout=True)
###
###ax = fig.add_subplot(111, projection='3d')
###cmap = mpl.cm.jet
###for s in range(2):
###    for t in range(n_time):
###        ax.scatter(pred_mean[0, t, s], pred_mean[1, t, s], pred_mean[2, t, s],
###                   s=t/3+1, color=cmap(s / float(2)))
###plt.show()
##
##'''Ploting the amplitude of the 1rst and 2nd components of each parameter 
##across time'''
##c1, c2, c3 = 0, 1, 2
##time = np.arange(n_time)
##fig = plt.figure(figsize=(16, 7))
##fig.suptitle('gng time ' + str(gng_time) + ' max delay ' + str(delay_max) +
##             ' parametrization ' + str(lamb) +
##             '\nGo / NoGo data projected onto dPCA decoder axis')
##
##plt.subplots_adjust(hspace = 0.5)
##plt.subplot(331)
##for s in range(2):
##    plt.plot(time, Z['t'][c1, :, s])
##plt.title(str(c1+1)+'st time component')
##
##
##plt.subplot(334)
##for s in range(2):
##    plt.plot(time, Z['t'][c2, :, s])
##plt.title(str(c2+1)+'d time component')
##plt.xlabel('time')
##
##plt.subplot(337)
##for s in range(2):
##    plt.plot(time, Z['t'][c2, :, s])
##plt.title(str(c3+1)+'d time component')
##plt.xlabel('time')
##
##plt.subplot(332)
##for s in range(2):
##    plt.plot(time, Z['s'][c1, :, s])
##plt.title(str(c1+1)+'st stimulus component')
##
##plt.subplot(335)
##for s in range(2):
##    plt.plot(time, Z['s'][c2, :, s])
##plt.title(str(c2+1)+'nd stimulus component')
##plt.xlabel('time')
##
##plt.subplot(338)
##for s in range(2):
##    plt.plot(time, Z['s'][c2, :, s])
##plt.title(str(c3+1)+'nd stimulus component')
##plt.xlabel('time')
##
##plt.subplot(333)
##for s in range(2):
##    plt.plot(time, Z['ts'][c1, :, s])
##plt.title(str(c1+1)+'st mixing component')
##
##plt.subplot(336)
##for s in range(2):
##    plt.plot(time, Z['ts'][c2, :, s])
##plt.title(str(c2+1)+'nd mixing component')
##plt.xlabel('time')
##plt.show()
##
##plt.subplot(339)
##for s in range(2):
##    plt.plot(time, Z['ts'][c2, :, s])
##plt.title(str(c3+1)+'nd mixing component')
##plt.xlabel('time')
##plt.show()
##
##plt.savefig(os.path.join(fig_dir, 'GoNogotime.png'))
##
##'''Ploting the 2 firsts demixed components for time, stimulus, and mixing'''
##
##
##fig = plt.figure(figsize=(15, FIG_HEIGHT), tight_layout=True)
##fig.suptitle('gng time ' + str(gng_time) + ' max delay ' + str(delay_max) +
##             ' parametrization ' + str(lamb) +
##             '\nGo / NoGo dPCA components')
##
##plt.subplot(131)
##for s in range(2):
##    plt.scatter(Z['t'][c1, :, s], Z['t'][c2, :, s], s=2*time+1)
###plot(E['t'], c='k', lw=5)
##plt.xlabel(str(c1+1)+'st time component')
##plt.ylabel(str(c2+1)+'nd time component')
##
##plt.subplot(132)
##for s in range(2):
##    plt.scatter(Z['s'][c1, :, s], Z['s'][c2, :, s], s=2*time+1)
###plot(E['s'], c='k', lw=5)
##plt.xlabel(str(c1+1)+'st stimulus component')
##plt.ylabel(str(c2+1)+'nd stimulus component')
##
##plt.subplot(133)
##for s in range(2):
##    plt.scatter(Z['ts'][c1, :, s], Z['ts'][c2, :, s], s=2*time+1)
###plot(E['s'], c='k', lw=5)
##plt.xlabel(str(c1+1)+'st mixing component')
##plt.ylabel(str(c2+1)+'nd mixing component')
##plt.show()
##
##plt.savefig(os.path.join(fig_dir, 'GoNogocomp.png'))
##
###significance_masks = dpca.significance_analysis(pred_mean, predictions,
###                                                axis='t', n_shuffles=10,
###                                                n_splits=10, n_consecutive=10)
###
###for s in range(2):
###    plt.scatter(Z['s'][0, s], Z['s'][1, s])
####plt.xlabel('1st stimulus component')
####plt.ylabel('2nd stimulus component')
###plt.show()