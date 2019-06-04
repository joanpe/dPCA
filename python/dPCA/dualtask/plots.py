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
PATH_SAVE = '/home/joanpe/dPCA/python/dPCA/dualtask/'
PATH_LOAD = '/home/joan/cluster_home/dPCA/python/dPCA/dualtask/'
sys.path.insert(0, PATH_LOAD)

# Noise range for the input to the RNN
noise_rng = np.array([0.2])
noise = noise_rng[0]
#noise_rng = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
# Time of appearence of the go- no go task. 0 for no task. if gng_rng = [-1] 
# then it runs a ramdom trial either of the dualtask, dpa alone or gng alone.
gng_rng = np.array(-1)
gng = gng_rng[0]
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

load_dir = os.path.join(PATH_LOAD, 'data_trainedwithnoise')
save_dir = os.path.join(PATH_SAVE, 'data')

data = np.load(os.path.join(load_dir, 'data_' + str(gng) + '_'
                                      + str(l) + '_' + str(delay)
                                      + '_i' + str(INST) + '_n' + str(noise_rng[0])
                                      + '-' + str(noise_rng[-1])
                                      + '_neu' + str(num_neurons[0])
                                      + '-' + str(num_neurons[-1]) + '.npz'))


#Plot example trials of the RNN
def _setup_visualizations(self):
    '''See docstring in RecurrentWhisperer.'''
    FIG_WIDTH = self.n_plot  # inches
    FIX_HEIGHT = 9  # inches
    self.fig = plt.figure(figsize=(FIG_WIDTH, FIX_HEIGHT),
                          tight_layout=True)

def plot_trials(self, data, start_time=0, stop_time=None):
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

    for trial_idx in range(self.n_plot):
        plt.subplot(self.n_plot/6, self.n_plot/6, trial_idx+1)
        if self.n_plot == 1:
            plt.title('Example trial', fontweight='bold')
        else:
            if gng_time==-1:
                if task_type[trial_idx] == 0:
                    plt.title('Dual-task | Acc DPA %d | Acc GNG %d' %
                              (ev_acc_dpa_dual,
                               ev_acc_gng_dual), fontweight='bold')
                elif task_type[trial_idx] == 1:
                    plt.title('DPA task | Acc DPA %d' %
                              (ev_acc_dpa_dpa),
                              fontweight='bold')
                else:
                    plt.title('GNG task | Acc GNG %d' %
                              (ev_acc_gng_gng),
                              fontweight='bold')
            else:
                plt.title('Example trial %d | Acc %d' % (trial_idx + 1,
                                                         ev_acc_dpa),
                            fontweight='bold')

        self._plot_single_trial(
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

@staticmethod
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
for inst in range(INST):
    self.n_plot = 36
    if gng_time==-1:
        task_type = data['task_choice']
    else:
        task_type = 0

    f = plt.figure(self.fig.number)
    plt.clf()
#TODO define inputs etc
    inputs = data['inputs']
    output = data['output']
    pred_output = data['pred_out']
    ev_acc_dpa = data['acc'][0]
    ev_acc_dpa_dual = predictions['ev_acc_dpa_dual']
    ev_acc_gng_dual = predictions['ev_acc_gng_dual']
    ev_acc_dpa_dpa = predictions['ev_acc_dpa_dpa']
    ev_acc_gng_gng = predictions['ev_acc_gng_gng']
    f = self.plot_trials(data)
    
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



#Accuracy across training

plt.figure()
label_added = False
for i in range(INST):
    data_acc = np.load(PATH + '/logs_-1/lamb0.0/noise0.2/delay0/neurons64/inst'
                   + str(i) + '/cdd99a83bc/accuracies.npz')
    acc_dpa = data_acc['acc_dpa']
    acc_gng = data_acc['acc_gng']
    acc_dpa_dual = data_acc['acc_dpa_dual']
    acc_gng_dual = data_acc['acc_gng_dual']
    acc_dpa_dpa = data_acc['acc_dpa_dpa']
    acc_gng_gng = data_acc['acc_gng_gng']
    n_epochs = data_acc['n_epochs']
    
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

fig_dir = os.path.join(PATH, 'data_trainedwithnoise')
fig.savefig(os.path.join(fig_dir, 'acc_across_train_inst' + str(INST) +
                         '_noise_' + str(noise) + '.svg'))
plt.close()

#Plot mean acc across training
acc_dpa_dual = []
acc_gng_dual = []
acc_dpa_dpa = []
acc_gng_gng = []
n_epochs = []

for i in range(INST):
    data_acc = np.load(PATH + '/logs_-1/lamb0.0/noise0.2/delay0/neurons64/inst'
                   + str(i) + '/cdd99a83bc/accuracies.npz')
    acc = data_acc['acc_dpa_dual']
    acc_dpa_dual.append(acc)
    acc = data_acc['acc_gng_dual']
    acc_gng_dual.append(acc)
    acc = data_acc['acc_dpa_dpa']
    acc_dpa_dpa.append(acc)
    acc = data_acc['acc_gng_gng']
    acc_gng_gng.append(acc)
    n = data_acc['n_epochs']
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
plt.plot(epochs, acc_dpa_dualmean, label='Dual DPA', color='r')
plt.plot(epochs, acc_gng_dualmean, label='Dual GNG', color='b')
plt.plot(epochs, acc_dpa_dpamean, label='DPA DPA', color='g')
plt.plot(epochs, acc_gng_gngmean, label='GNG GNG', color='cyan')

#plt.xlim([0, 100])
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Mean accuracy')
fig = plt.gcf()
#plt.show()


fig.savefig(os.path.join(fig_dir, 'mean_acc_across_train' + str(noise) + '.svg'))
plt.close()


# Plot accuracy of DPA in dual task against accuracy of DPA in dpa alone

dual_acc = data['acc'][0][3]
dpa_acc = data['acc'][0][5]
n = np.arange(INST)

f, ax = plt.subplots()
ax.scatter(dpa_acc, dual_acc)
ax.plot([0.4, 1], [0.4, 1], ls='--', color='grey')
plt.xlabel('DPA acc')
plt.ylabel('dual DPA acc')
f.savefig(os.path.join(fig_dir, 'dpa_vs_dual_acc' + str(noise) + '.svg'))

# Plot accuracy across training of the instances where acc DPA > acc DPA 
# dual at the end of the training and also acc dpa>0.55

acc_dpa_dual = []
acc_dpa_dpa = []
n_epochs = []
inst_cond = []
for i in range(INST):
    data = np.load(PATH + '/logs_-1/lamb0.0/noise0.2/delay0/neurons64/inst'
                   + str(i) + '/cdd99a83bc/accuracies.npz')
    if dpa_acc[i] > 0.55:
        if dual_acc[i] < dpa_acc[i]:
            acc = data['acc_dpa_dual']
            acc_dpa_dual.append(acc)
            acc = data['acc_dpa_dpa']
            acc_dpa_dpa.append(acc)
            n = data['n_epochs']
            n_epochs.append(n)
            inst_cond.append(i)
            
#inst_cond = np.array(inst_cond)
n = np.shape(n_epochs)[0]
min_epochs = np.min(tuple(n_epochs[i] for i in range(n)))//10
acc_dpa_dualstack = acc_dpa_dual[0][0:min_epochs]
acc_dpa_dpastack = acc_dpa_dpa[0][0:min_epochs]
for i in range(n-1):
    acc_dpa_dualstack = np.column_stack((acc_dpa_dualstack,
                                         acc_dpa_dual[i+1][0:min_epochs]))
    acc_dpa_dpastack = np.column_stack((acc_dpa_dpastack,
                                        acc_dpa_dpa[i+1][0:min_epochs]))

acc_dpa_dualmean = np.mean(acc_dpa_dualstack, axis=1)
acc_dpa_dpamean = np.mean(acc_dpa_dpastack, axis=1)

acc_dpa_dualstd = np.std(acc_dpa_dualstack, axis=1)
acc_dpa_dpastd = np.std(acc_dpa_dpastack, axis=1)

epochs = np.arange(min_epochs)

plt.figure()
plt.plot(epochs, acc_dpa_dualmean, label='Dual DPA', color='r')
plt.plot(epochs, acc_dpa_dpamean, label='DPA DPA', color='g')

for i, num in enumerate(inst_cond):
    plt.plot(epochs, acc_dpa_dualstack[:, i], color='r', alpha=0.2)
    plt.plot(epochs, acc_dpa_dpastack[:, i], color='g', alpha=0.2)
    plt.annotate(num, (epochs[-1] + 1, acc_dpa_dualstack[-1, i]))
#plt.fill_between(epochs, acc_dpa_dualmean-acc_dpa_dualstd, acc_dpa_dualmean+
#                 acc_dpa_dualstd, color='r', alpha=0.5)
#plt.fill_between(epochs, acc_dpa_dpamean-acc_dpa_dpastd, acc_dpa_dpamean+
#                 acc_dpa_dpastd, color='g', alpha=0.5)

#plt.xlim([0, 100])
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Mean accuracy')
fig = plt.gcf()
plt.show()


fig.savefig(os.path.join(fig_dir, 'mean_acc_across_train_dual_vs_dpa' + str(noise) + '.svg'))
plt.close()