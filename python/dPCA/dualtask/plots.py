#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 09:18:34 2019

@author: joan
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import random
# from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
PATH_SAVE = '/home/joan/dPCA/python/dPCA/dualtask/'
PATH_LOAD = '/home/joan/cluster_home/dPCA/python/dPCA/dualtask/'
sys.path.insert(0, PATH_LOAD)

# Noise range for the input to the RNN
noise_rng = np.array([0.2])
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
# number of RNN instances
INST = 50
#Number of example plots to show
n_plot = 36

load_dir = os.path.join(PATH_LOAD, 'data_trainedwithnoise')
save_dir = os.path.join(PATH_SAVE, 'data')

data = np.load(os.path.join(load_dir, 'data_' + str(gng) + '_'
                                      + str(l) + '_' + str(delay)
                                      + '_i' + str(INST) + '_n' + str(noise_rng[0])
                                      + '-' + str(noise_rng[-1])
                                      + '_neu' + str(num_neurons[0])
                                      + '-' + str(num_neurons[-1]) + '.npz'))


#Plot example trials of the RNN

def plot_trials(inputs, output, pred_output, vec_acc_dpa_dpa, vec_acc_dpa_dual,
                task_type, start_time=0, stop_time=None):
    '''Plots example trials, complete with input pulses, correct outputs,
    and RNN-predicted outputs.

    Args:
        data: dict as returned by generate_dualtask_trials.

        start_time (optional): int specifying the first timestep to plot.
        Default: 0.

        stop_time (optional): int specifying the last timestep to plot.
        Default: n_time.

    Returns:
        None.
    '''
    hps = {
        'rnn_type': 'vanilla',
        'n_hidden': neuron,
        'min_loss': 1e-6,  # 1e-4
        'min_learning_rate': 1e-5,
        'max_n_epochs': 5000,
        'do_restart_run': True,
        'log_dir': './logs_' + str(gng) + '/lamb' + str(lamb) + '/noise' +
        str(noise) + '/delay' + str(delay) + '/neurons' +
        str(neuron) + '/inst' + str(inst),
        'data_hps': {
            'n_batch': 2048,
            'n_time': 20,
            'n_bits': 6,
            'noise': noise,
            'gng_time': gng,
            'lamb': lamb,
            'delay_max': delay_max},
        }
    n_time = hps.data_hps['n_time']
    gng_time = hps.data_hps['gng_time']
#        n_plot = np.min([hps.n_trials_plot, n_batch])
    if stop_time is None:
        stop_time = n_time

    time_idx = range(start_time, stop_time)

    for trial_idx in range(n_plot):
        plt.subplot(n_plot/6, n_plot/6, trial_idx+1)
        if n_plot == 1:
            plt.title('Example trial', fontweight='bold')
        else:
            if gng_time==-1:
                if task_type[trial_idx] == 0:
                    plt.title('Dual-task | Acc DPA %d | Pred %.4e | Out %.2e' %
                                  (vec_acc_dpa_dual[np.where(np.where(task_type==0)[0]==trial_idx)[0]],
                                   pred_output[trial_idx, n_time-1, 0],
                                   output[trial_idx, n_time-1, 0]), fontweight='bold')
                elif task_type[trial_idx] == 1:
                    plt.title('DPA task | Acc DPA %d | Pred %.4e | Out %.2e' %
                              (vec_acc_dpa_dpa[np.where(np.where(task_type==1)[0]==trial_idx)[0]],
                               pred_output[trial_idx, n_time-1, 0],
                               output[trial_idx, n_time-1, 0]),
                              fontweight='bold')
            else:
                plt.title('Example trial %d | Acc %d' % (trial_idx + 1,
                                                         ev_acc_dpa),
                            fontweight='bold')

        _plot_single_trial(
            inputs[trial_idx, time_idx, :],
            output[trial_idx, time_idx, :],
            pred_output[trial_idx, time_idx, :])

        # Only plot x-axis ticks and labels on the bottom subplot
        if trial_idx < (n_plot-1):
            plt.xticks([])
        else:
            plt.xlabel('Timestep', fontweight='bold')

    f = plt.gcf()
#        plt.ion()
#        plt.show()
#        plt.pause(1e-10)
    return f


def _plot_single_trial(input_txd, output_txd, pred_output_txd):

    VERTICAL_SPACING = 2.5
    [n_time, n_bits] = input_txd.shape
    tt = range(n_time)

    y_ticks = [VERTICAL_SPACING*bit_idx for bit_idx in range(n_bits)]
    y_tick_labels = ['S %d' % (bit_idx+1) for bit_idx in range(n_bits)]
    plt.yticks(y_ticks, y_tick_labels, fontweight='bold')
    for bit_idx in range(n_bits):

        vertical_offset = VERTICAL_SPACING*bit_idx

        # Input pulses
        plt.fill_between(
            tt,
            vertical_offset + input_txd[:, bit_idx],
            vertical_offset,
            step='mid',
            color='gray')

        # Correct outputs
        plt.step(
            tt,
            vertical_offset + output_txd[:, bit_idx],
            where='mid',
            linewidth=2,
            color='cyan')

        if bit_idx == 0:
            # RNN outputsp
            plt.step(
                tt,
                vertical_offset + pred_output_txd[:, 0],
                where='mid',
                color='purple',
                linewidth=1.5,
                linestyle='--')

    plt.xlim(-1, n_time)
    plt.ylim(-1, n_bits*2+2)





# Plot example trials

FIG_WIDTH = n_plot  # inches
FIX_HEIGHT = 9  # inches
fig = plt.figure(figsize=(FIG_WIDTH, FIX_HEIGHT),
                      tight_layout=True)   

for inst in range(INST):
    if gng==-1:
        task_type = data['task_choice'][inst]
    else:
        task_type = 0

    f = plt.figure(fig.number)
    plt.clf()

    inputs = data['inputs'][inst]
    output = data['output'][inst]
    pred_output = data['pred_out'][inst]
    ev_acc_dpa = data['acc'][0][1][inst]
    ev_acc_dpa_dual = data['acc'][0][3][inst]
    ev_acc_gng_dual = data['acc'][0][4][inst]
    ev_acc_dpa_dpa = data['acc'][0][5][inst]
    ev_acc_gng_gng = data['acc'][0][6][inst]
    vec_acc_dpa_dual = data['acc'][0][7][inst]
    vec_acc_dpa_dpa = data['acc'][0][8][inst]
    f = plot_trials(inputs, output, pred_output, vec_acc_dpa_dpa,
                    vec_acc_dpa_dual, task_type)
    
    plot_dir = os.path.join(save_dir, 'task_plots/noise' + str(noise) +
                            'lamb' + str(l))
    if os.path.isdir(plot_dir) is False:
        os.mkdir(plot_dir)
        f.savefig(os.path.join(plot_dir, 'Inst' + str(inst) + '.svg'))
    else:
        f.savefig(os.path.join(plot_dir, 'Inst' + str(inst) + '.svg'))
    
    plt.close()

'''Plot mean data together'''
## Plots for number of neurons against accuracy
## Loading the data for task without distractor
#plt.figure()
#
#for l in lamb:
#    for delay in delay_max:
#        data = np.load(fig_dir + '/data_0_' + str(l) + '_' + str(delay) +
#                       '_i' + str(INST) + '_n' + str(noise_rng[0]) + '-' +
#                       str(noise_rng[-1]) + '_neu' + str(num_neurons[0])
#                       + '-' + str(num_neurons[-1]) + '.npz')
#        data = data['acc']
#        
#        # Compute the mean accuracy across instances
#        mean_acc = []
#        std = []
#        for n in range(num_neurons.shape[0]):
#            mean_acc.append(np.mean(data[n][1]))
#            std.append(np.std(data[n][1]))
#            plt.scatter(np.repeat(num_neurons[n], INST), data[n][1], marker='.', color='b')
#       
#        # Plot with error bars of the accuracy / loss
#        plt.plot(num_neurons, mean_acc, marker='+', ms=15, color='b',
#                     label='DPA accuracy dpa gng0 lamb' + str(l))
#
## Loading data for task with distractor
#for l in lamb:
#    for delay in delay_max:
#        data10 = np.load(fig_dir + '/data_10_' + str(l) + '_' + str(delay) +
#                         '_i' + str(INST) + '_n' + str(noise_rng[0]) + '-' +
#                         str(noise_rng[-1]) + '_neu' + str(num_neurons[0])
#                         + '-' + str(num_neurons[-1]) + '.npz')
#        data10 = data10['acc']
#
#        # Compute the mean accuracy across instances
#        mean_acc = []
#        std = []
#        for n in range(num_neurons.shape[0]):
#            mean_acc.append(np.mean(data10[n][1]))
#            std.append(np.std(data10[n][1]))
#            plt.scatter(np.repeat(num_neurons[n], INST), data10[n][1], marker='.', color='r')
#            
#        # Plot with error bars of the accuracy / loss
#        plt.plot(num_neurons, mean_acc, marker='+', ms=15, color='r',
#                     label='DPA accuracy gng10 lamb' + str(l))
#
#plt.xlabel('Number of neurons')
#plt.ylabel('Mean accuracy')
#plt.legend()
#plt.show()
#
#if os.path.isdir(fig_dir) is False:
#    os.mkdir(fig_dir)
#    plt.savefig(os.path.join(fig_dir, 'mean_acc_neurons_'
#                             + str(l) + '_' + str(delay)
#                             + '_i' + str(INST) + '_n' + str(noise_rng[0])
#                             + '-' + str(noise_rng[-1]) + '.png'))
#else:
#    plt.savefig(os.path.join(fig_dir, 'mean_acc_neurons_'
#                             + str(l) + '_' + str(delay)
#                             + '_i' + str(INST) + '_n' + str(noise_rng[0])
#                             + '-' + str(noise_rng[-1]) + '.png'))

#
## Plots for lambda against accuracy
## Loading the data for both tasks
#f = plt.figure()
#mean_acc_dpa0 = []
#mean_acc_gng0 = []
#std_dpa0 = []
#std_gng0 =[]
#mean_acc_dpa10 = []
#mean_acc_gng10 = []
#std_dpa10 = []
#std_gng10 =[]
#for gng in gng_rng:
#    for l in lamb:
#        for delay in delay_max:
#            data = np.load(fig_dir + '/data_' + str(gng) + '_' + str(l) + '_'
#                           + str(delay) + '_i' + str(INST) + '_n'
#                           + str(noise_rng[0]) + '-' + str(noise_rng[-1])
#                           + '_neu' + str(num_neurons[0])
#                           + '-' + str(num_neurons[-1]) + '.npz')
#            datal = data['acc'][0]
##            datal.append(data['acc'])
#            
#            # Compute the mean accuracy across instances
#        if gng > 0:
#            mean_acc_dpa10.append(np.mean(datal[1]))
#            std_dpa10.append(np.std(datal[1]))
#            mean_acc_gng10.append(np.mean(datal[2]))
#            std_gng10.append(np.std(datal[2]))
#        else:
#            mean_acc_dpa0.append(np.mean(datal[1]))
#            std_dpa0.append(np.std(datal[1]))
#            mean_acc_gng0.append(np.mean(datal[2]))
#            std_gng0.append(np.std(datal[2]))
#
## Plot with error bars of the accuracy / loss
##        plt.errorbar(l, mean_acc_dpa, yerr=std_dpa, marker='+',
##                     label='DPA with gng' + str(gng))
##plt.plot(lamb, mean_acc_dpa10, color='r', label='DPA with distractor')
#plt.errorbar(lamb+0.02, mean_acc_dpa10, yerr=std_dpa10, marker='+', color='r', label='DPA with distractor')
##        plt.errorbar(l, mean_acc_gng, yerr=std_gng, marker='+',
##                     label='GNG with gng' + str(gng))
##plt.plot(lamb, mean_acc_dpa0, color='g', label='DPA no distractor')
#plt.errorbar(lamb-0.02, mean_acc_dpa0, yerr=std_dpa0, marker='+', color='g', label='DPA no distractor')
#
##plt.plot(lamb, mean_acc_gng10, color='k', label='GNG')
#plt.errorbar(lamb, mean_acc_gng10, yerr=std_gng10, marker='+', color='k', label='GNG')
#
#plt.xlabel('Parametrization')
#plt.ylabel('Mean accuracy')
#plt.legend()
#plt.show()
#
#if os.path.isdir(fig_dir) is False:
#    os.mkdir(fig_dir)
#    plt.savefig(os.path.join(fig_dir, 'mean_ acc_lambda_'
#                             + str(l) + '_' + str(delay)
#                             + '_i' + str(INST) + '_n' + str(noise_rng[0])
#                             + '-' + str(noise_rng[-1]) + '_neu'
#                             + str(num_neurons[0]) + '-'
#                             + str(num_neurons[-1]) + '.png'))
#else:
#    plt.savefig(os.path.join(fig_dir, 'mean_acc_lambda_'
#                             + str(l) + '_' + str(delay)
#                             + '_i' + str(INST) + '_n' + str(noise_rng[0])
#                             + '-' + str(noise_rng[-1]) + '_neu'
#                             + str(num_neurons[0]) + '-'
#                             + str(num_neurons[-1]) + '.png'))

#
## Plots for noise against accuracy
## Loading data for both tasks
#f = plt.figure()
#for gng in gng_rng:
#    for l in lamb:
#        for delay in delay_max:
#            data10 = np.load(fig_dir + '/data_' + str(gng) + '_' + str(l)
#                            + '_' + str(delay) + '_i' + str(INST) + '_n'
#                            + str(noise_rng[0]) + '-' +
#                             str(noise_rng[-1]) + '_neu' + str(num_neurons[0])
#                           + '-' + str(num_neurons[-1]) + '.npz')
#            data10 = data10['acc']
#        
#            # Compute the mean accuracy across instances
#            mean_acc = []
#            std = []
#            for n in range(noise_rng.shape[0]):
#                mean_acc.append(np.mean(data10[n][1]))
#                std.append(np.std(data10[n][1]))
#        
#            # Plot with error bars of the accuracy / loss
#            plt.errorbar(noise_rng, mean_acc, yerr=std, marker='+',
#                         label='DPA accuracy dpa gng' + str(l))
#
#
#plt.xlabel('Noise')
#plt.ylabel('Mean accuracy')
#plt.legend()
#plt.show()
#
#if os.path.isdir(fig_dir) is False:
#    os.mkdir(fig_dir)
#    plt.savefig(os.path.join(fig_dir, 'mean_ acc_noise_'
#                             + str(l) + '_' + str(delay)
#                             + '_i' + str(INST) + '_n' + str(noise_rng[0])
#                             + '-' + str(noise_rng[-1]) + '_neu'
#                             + str(num_neurons[0]) + '-'
#                             + str(num_neurons[-1]) + '.png'))
#else:
#    plt.savefig(os.path.join(fig_dir, 'mean_acc_noise_'
#                             + str(l) + '_' + str(delay)
#                             + '_i' + str(INST) + '_n' + str(noise_rng[0])
#                             + '-' + str(noise_rng[-1]) + '_neu'
#                             + str(num_neurons[0]) + '-'
#                             + str(num_neurons[-1]) + '.png'))
#
#


# Plots for accuracy depending on task.

# General accuracy



# Plots for accuracy depending on task.


#Accuracy across training
noise = noise_rng[0]
plt.figure()
label_added = False
for i in range(INST):
    data = np.load(PATH_LOAD + '/logs_-1/lamb0.0/noise' + str(noise) +
                   '/delay0/neurons64/inst' + str(i) +
                   '/9afbb8777a/accuracies.npz')
    
    acc_dpa = data['acc_dpa']
    acc_gng = data['acc_gng']
    acc_dpa_dual = data['acc_dpa_dual']
    acc_gng_dual = data['acc_gng_dual']
    acc_dpa_dpa = data['acc_dpa_dpa']
    acc_gng_gng = data['acc_gng_gng']
    n_epochs = data['n_epochs']
    
    epochs = np.arange(n_epochs//10)
    if not label_added:
        plt.plot(epochs, acc_dpa_dual, label='Dual DPA', color='r')
        plt.plot(epochs, acc_gng_dual, label='Dual GNG', color='b')
        plt.plot(epochs, acc_dpa_dpa, label='DPA DPA', color='g')
        plt.plot(epochs, acc_gng_gng, label='GNG GNG', color='cyan')
        label_added = True        
    else:
        plt.plot(epochs, acc_dpa_dual, color='r')
        plt.plot(epochs, acc_gng_dual, color='b')
        plt.plot(epochs, acc_dpa_dpa, color='g')
        plt.plot(epochs, acc_gng_gng, color='cyan')

#plt.xlim([0, 100])
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
fig = plt.gcf()
# plt.show()

fig.savefig(os.path.join(save_dir, 'acc_across_train_inst' + str(INST) +
                         '_noise_' + str(noise) + '.png'))
plt.close()


#Plot for the mean accuracy of DPA and dual with all trials

acc_dpa_dual = []
acc_gng_dual = []
acc_dpa_dpa = []
acc_gng_gng = []
n_epochs = []

for i in range(INST):
    #TDO find as before
    data = np.load(PATH_LOAD + '/logs_-1/lamb0.0/noise' + str(noise) +
                   '/delay0/neurons64/inst' + str(i) +
                   '/9afbb8777a/accuracies.npz')
    acc = data['acc_dpa_dual']
    acc_dpa_dual.append(acc)
    acc = data['acc_gng_dual']
    acc_gng_dual.append(acc)
    acc = data['acc_dpa_dpa']
    acc_dpa_dpa.append(acc)
    acc = data['acc_gng_gng']
    acc_gng_gng.append(acc)
    n = data['n_epochs']
    n_epochs.append(n)


min_epochs = np.min(tuple(n_epochs[i] for i in range(INST)))//10
acc_dpa_dualstack = acc_dpa_dual[0][0:min_epochs]
acc_gng_dualstack = acc_gng_dual[0][0:min_epochs]
acc_dpa_dpastack = acc_dpa_dpa[0][0:min_epochs]
acc_gng_gngstack = acc_gng_gng[0][0:min_epochs]
for i in range(INST-1):
    acc_dpa_dualstack = np.column_stack((acc_dpa_dualstack,
                                         acc_dpa_dual[i+1][0:min_epochs]))
    acc_gng_dualstack = np.column_stack((acc_gng_dualstack,
                                         acc_gng_dual[i+1][0:min_epochs]))
    acc_dpa_dpastack = np.column_stack((acc_dpa_dpastack,
                                        acc_dpa_dpa[i+1][0:min_epochs]))
    acc_gng_gngstack = np.column_stack((acc_gng_gngstack,
                                        acc_gng_gng[i+1][0:min_epochs]))

acc_dpa_dualmean = np.mean(acc_dpa_dualstack, axis=1)
acc_gng_dualmean = np.mean(acc_gng_dualstack, axis=1)
acc_dpa_dpamean = np.mean(acc_dpa_dpastack, axis=1)
acc_gng_gngmean = np.mean(acc_gng_gngstack, axis=1)

epochs = np.arange(min_epochs)

plt.figure()
plt.plot(epochs, acc_dpa_dualmean, label='Dual DPA', color='r', linewidth=3)
plt.plot(epochs, acc_dpa_dpamean, label='DPA DPA', color='g', linewidth=3)
for i in range(INST):
    plt.plot(epochs, acc_dpa_dualstack[:, i], color='r', alpha=0.1)
    plt.plot(epochs, acc_dpa_dpastack[:, i], color='g', alpha=0.1)

#plt.xlim([0, 100])
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Mean accuracy')
fig = plt.gcf()
#plt.show()


fig.savefig(os.path.join(save_dir, 'mean_acc_across_train' + str(noise) + '.png'))
plt.close()



# Plot accuracy of DPA in dual task against accuracy of DPA in dpa alone with numbers
data = np.load(os.path.join(load_dir, 'data_-1_0.0_0_i50_n' + str(noise) + '-' +
                            str(noise) + '_neu64-64.npz'))

dual_acc = data['acc'][0][3]
dpa_acc = data['acc'][0][5]
n = np.arange(INST)

f, ax = plt.subplots()
ax.scatter(dpa_acc, dual_acc, color='b', s=5)
ax.plot([0.4, 1], [0.4, 1], ls='--', color='grey')
for i, num in enumerate(n):
    plt.annotate(num, (dpa_acc[i], dual_acc[i]))
plt.xlabel('DPA acc')
plt.ylabel('dual DPA acc')
f.savefig(os.path.join(save_dir, 'dpa_vs_dual_accnumber_' + str(noise) + '.png'))

# Plot accuracy of DPA in dual task against accuracy of DPA in dpa alone without numbers
data = np.load(os.path.join(PATH_LOAD, 'data_-1_0.0_0_i50_n' + str(noise) + '-' +
                            str(noise) + '_neu64-64.npz'))

dual_acc = data['acc'][0][3]
dpa_acc = data['acc'][0][5]
n = np.arange(INST)

f, ax = plt.subplots()
ax.scatter(dpa_acc, dual_acc, color='b', s=5)
ax.plot([0.4, 1], [0.4, 1], ls='--', color='grey')
plt.xlabel('DPA acc')
plt.ylabel('dual DPA acc')
f.savefig(os.path.join(save_dir, 'dpa_vs_dual_acc_' + str(noise) + '.png'))


# Bar plot acc dual vs acc dpa

plt.figure()
x = np.arange(2)
means = [np.mean(dual_acc), np.mean(dpa_acc)]
plt.bar(x, means, color='b', width=0.3)
plt.xticks(x, ('Dual-task', 'DPA task'))
plt.title('Mean accuracy')
plt.savefig(os.path.join(save_dir, 'mean_acc_bar' + str(noise) + '.png'))
plt.close('all')




# Count which number of stimulus pairs (s1-s3/s4 or s2-s3/s4) are correct
# for the conditions that appears s5 or s6 during the distractor

data = np.load(os.path.join(load_dir, 'data_-1_0.0_0_i50_n' + str(noise) + '-' +
                        str(noise) + '_neu64-64.npz'))

task = data['task']
stim = data['stim_conf']
stim_dual = []
stim_dpa = []
acc_dual = []
acc_dpa = []
for i in range(INST):
    stim_dual.append(stim[i, task[i, :]==0])
    stim_dpa.append(stim[i, task[i, :]==1])
    acc_dual.append(data['acc'][0][7][i]*1)
    acc_dpa.append(data['acc'][0][8][i]*1)

matdual_inst = []
matdpa_inst = []
for i in range(INST):
    matdual = np.zeros((2, 2, 2))
    for gng in range(2):
        matdpa = np.zeros((2, 2))
        for dpa1 in range(2):
            for dpa2 in range(2):
                ind_dual = np.logical_and.reduce((stim_dual[i][:, 0]==dpa1,
                                                stim_dual[i][:, 1]==dpa2,
                                                stim_dual[i][:, 2]==gng))
                matdual[dpa1, dpa2, gng] = np.sum(acc_dual[i][ind_dual])
                ind_dpa = np.logical_and(stim_dpa[i][:, 0]==dpa1,
                                         stim_dpa[i][:, 1]==dpa2)
                matdpa[dpa1, dpa2] = np.sum(acc_dpa[i][ind_dpa])
    
    
    
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(9.75, 3))    

    plt.subplots_adjust(wspace=0.4)

#    plt.subplot(2, 2, 1).set_title('Stimulus S5 appears')
    im = ax[0].imshow(matdual[:, :, 0], cmap='GnBu', vmin=0, vmax=180)
    ax[0].set_title('Stimulus S5 appears')
    ax[0].set_xticklabels(['', 'S3', 'S4', ''])
    ax[0].set_yticklabels(['', 'S1', '', 'S2', ''])
#    fig.colorbar(im)
    
#    plt.subplot(2, 2, 3).set_title('Stimulus S6 appears')
    im2 = ax[1].imshow(matdual[:, :, 1], cmap='GnBu', vmin=0, vmax=180)
    ax[1].set_title('Stimulus S6 appears')
#    plt.colorbar()
    ax[1].set_xticklabels(['', 'S3', 'S4', ''])
    ax[1].set_yticklabels(['', 'S1', '', 'S2', ''])
#    fig.colorbar(im2)
#    plt.subplot(2, 2, 2).set_title('No distractor')
    im3 = ax[2].imshow(matdpa, cmap='GnBu', vmin=0, vmax=180)
    ax[2].set_title('No distractor')
#    plt.colorbar()
    ax[2].set_xticklabels(['', 'S3', 'S4', ''])
    ax[2].set_yticklabels(['', 'S1', '', 'S2', ''])
    
    
    fig.colorbar(im, ax=ax.ravel().tolist(), shrink=0.7)
    
    
       
    
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)
        plt.savefig(os.path.join(plot_dir, 'Inst' + str(i) + '.png'))
    else:
        plt.savefig(os.path.join(plot_dir, 'Inst' + str(i) + '.png'))
    plt.close()
    
    matdual_inst.append(matdual)
    matdpa_inst.append(matdpa)


    