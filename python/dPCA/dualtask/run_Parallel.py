"""
Created on Thu May  2 18:55:13 2019

@author: joan
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import pdb
import sys
import os
import random
# from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
PATH = '/home/joanpe/dPCA/python/dPCA/dualtask/'
sys.path.insert(0, PATH) 
from DualTask import DualTask
#from dPCA import dPCA
from joblib import Parallel, delayed
import multiprocessing

# *****************************************************************************
# STEP 1: Train RNNs to solve the dual task *********************************
# *****************************************************************************
# Noise range for the input to the RNN
noise_rng = np.array([0.0])
#noise_rng = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
# Time of appearence of the go- no go task. 0 for no task. if gng_rng = [-1] 
# then it runs a ramdom trial either of the dualtask, dpa alone or gng alone.
gng_rng = np.array(-1)
#gng_rng = np.array([0, 10])
lamb = np.array([0.0])
#lamb = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1])
delay_max = np.array([0])
num_neurons = np.array([64])
# number of RNN instances
INST = 50


def trainDualTask(noise, gng, inst, lamb, delay, neuron):
    '''Train an RNN with a given noise and compute the value of the accuracy
    of its predictions'''
    # Hyperparameters for AdaptiveLearningRate
    alr_hps = {'initial_rate': 0.1}

    # Hyperparameters for DualTask
    # See DualTask.py for detailed descriptions.

    hps = {
        'rnn_type': 'vanilla',
        'n_hidden': neuron,
        'min_loss': 1e-6,  # 1e-4
        'min_learning_rate': 1e-5,
        'max_n_epochs': 5000,
        'do_restart_run': False,
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
        'alr_hps': alr_hps
        }

    # Create DualTask object
    dt = DualTask(**hps)
    # Train the RNN instance for the specific noise
    dt.train()

# Get inputs and outputs from example trials
    random.seed(noise*INST)
    example_trials = dt.generate_dualtask_trials()

    is_lstm = dt.hps.rnn_type == 'lstm'

    # Compute RNN predictions from example trials
    example_predictions = dt.predict(example_trials,
                                     do_predict_full_LSTM_state=is_lstm)

    # Accuracy of the predictions
    acc_dpa = example_predictions['ev_acc_dpa']
    acc_gng = example_predictions['ev_acc_gng']
    state = example_predictions['state']
    acc_dpa_dual = example_predictions['ev_acc_dpa_dual']
    acc_gng_dual = example_predictions['ev_acc_gng_dual']
    acc_dpa_dpa = example_predictions['ev_acc_dpa_dpa']
    acc_gng_gng = example_predictions['ev_acc_gng_gng']
    if gng == -1:
        task_type = example_trials['task_choice']
    else:
        task_type = 0


# Plot example trials
    f = dt.plot_trials(example_trials)

    plot_dir = os.path.join(PATH, 'task_plots')
    if os.path.isdir(plot_dir) is False:
        os.mkdir(plot_dir)
        f.savefig(os.path.join(plot_dir, 'Inst' + str(inst) + '.svg'))
    else:
        f.savefig(os.path.join(plot_dir, 'Inst' + str(inst) + '.svg'))

    plt.close()
    return [acc_dpa, acc_gng, state, task_type, acc_dpa_dual, acc_gng_dual,
            acc_dpa_dpa, acc_gng_gng]

# Condition for which we assign 1 different task of 3 in each trial
if gng_rng == -1:
    # Train various RNNs with diferent noise
    #for gng in gng_rng:
    gng = gng_rng
    acc = []
    state = []
    
    for l in  lamb:
        
        for delay in delay_max:
    
            for noise in noise_rng:
        
                for neuron in num_neurons:
                    numcores = multiprocessing.cpu_count()
                    ops = Parallel(n_jobs=numcores)(delayed(
                            trainDualTask)(noise, gng, inst, l, delay,
                                         neuron) for inst in range(INST))
                    
                    # Save data in a list
                    acc_dpa = []
                    acc_gng = []
                    task = []
                    acc_dpa_dual = []
                    acc_gng_dual = []
                    acc_dpa_dpa = []
                    acc_gng_gng = []
                    for i in range(INST):
                        acc_dpa.append(ops[i][0])
                        acc_gng.append(ops[i][1])
                        state.append([noise, ops[i][2]])
                        task.append(ops[i][3])
                        acc_dpa_dual.append(ops[i][4])
                        acc_gng_dual.append(ops[i][5])
                        acc_dpa_dpa.append(ops[i][6])
                        acc_gng_gng.append(ops[i][7])
                        
                    acc.append([noise, acc_dpa, acc_gng, acc_dpa_dual,
                                acc_gng_dual, acc_dpa_dpa, acc_gng_gng])
                  
    
        # save data and figure
            data = {'acc': acc, 'state': state, 'task': task}
        
            fig_dir = os.path.join(PATH, 'data_trainedwithnoise')
            if os.path.isdir(fig_dir) is False:
                os.mkdir(fig_dir)
                np.savez(os.path.join(fig_dir, 'data_' + str(gng) + '_'
                                      + str(l) + '_' + str(delay)
                                      + '_i' + str(INST) + '_n' + str(noise_rng[0])
                                      + '-' + str(noise_rng[-1])
                                      + '_neu' + str(num_neurons[0])
                                      + '-' + str(num_neurons[-1])), **data)
            else:
                np.savez(os.path.join(fig_dir, 'data_' + str(gng) + '_'
                                      + str(l) + '_' + str(delay)
                                      + '_i' + str(INST) + '_n' + str(noise_rng[0])
                                      + '-' + str(noise_rng[-1])
                                      + '_neu' + str(num_neurons[0])
                                      + '-' + str(num_neurons[-1])), **data)
#
## Runs DPA + GNG task
#else:
#    # Train various RNNs with diferent noise
#    for gng in gng_rng:
#        f = plt.figure()
#        plt.clf()
#        
#        for l in  lamb:
#            acc = []
#            state = []
#            
#            for delay in delay_max:
#        
#                for noise in noise_rng:
#            
#                    for neuron in num_neurons:
#                        numcores = multiprocessing.cpu_count()
#                        ops = Parallel(n_jobs=numcores)(delayed(
#                                trainDualTask)(noise, gng, inst, l, delay,
#                                             neuron) for inst in range(INST))
#                        
#                        # Save data in a list
#                        NOISE = np.repeat(l, INST)
#                        acc_dpa = []
#                        acc_gng = []
#                        task_type = []
#                        for i in range(INST):
#                            acc_dpa.append(ops[i][0])
#                            acc_gng.append(ops[i][1])
#                            state.append([noise, ops[i][2]])
#                                    
#                        acc.append([noise, acc_dpa, acc_gng])
#                        # Plot loss / accuracy for the different noise- instances
#                        plt.figure(f.number)
#                #        plt.plot(noise, loss_dpa, '+')
#                #        plt.plot(noise, loss_gng, 'v')
#                        plt.plot(NOISE, acc_dpa, '+', color='k')
##                                plt.plot(NOISE, acc_gng, 'v', color='k')
#                        plt.xlabel('Num neurons')
#                        plt.ylabel('Accuracy')
#                        plt.ion()
#                        plt.draw()
#                        plt.show()
#                        plt.pause(0.01)
#        
#            # save data and figure
#                data = {'acc': acc, 'state': state}
#            
#                fig_dir = os.path.join(PATH, 'data_trainedwithnoise')
#                if os.path.isdir(fig_dir) is False:
#                    os.mkdir(fig_dir)
#                    np.savez(os.path.join(fig_dir, 'data_' + str(gng) + '_'
#                                          + str(l) + '_' + str(delay)
#                                          + '_i' + str(INST) + '_n' + str(noise_rng[0])
#                                          + '-' + str(noise_rng[-1])
#                                          + '_neu' + str(num_neurons[0])
#                                          + '-' + str(num_neurons[-1])), **data)
#                    plt.savefig(os.path.join(fig_dir, 'acc_noise_' + str(gng) + '_'
#                                             + str(l) + '_' + str(delay)
#                                             + '_i' + str(INST) + '_n' + str(noise_rng[0])
#                                             + '-' + str(noise_rng[-1])
#                                             + '_neu' + str(num_neurons[0])
#                                             + '-' + str(num_neurons[-1]) + '.png'))
#                else:
#                    np.savez(os.path.join(fig_dir, 'data_' + str(gng) + '_'
#                                          + str(l) + '_' + str(delay)
#                                          + '_i' + str(INST) + '_n' + str(noise_rng[0])
#                                          + '-' + str(noise_rng[-1])
#                                          + '_neu' + str(num_neurons[0])
#                                          + '-' + str(num_neurons[-1])), **data)
#                    plt.savefig(os.path.join(fig_dir, 'acc_noise_' + str(gng) + '_'
#                                             + str(l) + '_' + str(delay)
#                                             + '_i' + str(INST) + '_n' + str(noise_rng[0])
#                                             + '-' + str(noise_rng[-1])
#                                             + '_neu' + str(num_neurons[0])
#                                             + '-' + str(num_neurons[-1]) + '.png'))


fig_dir = os.path.join(PATH, 'data_trainedwithnoise')

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
    data = np.load(PATH + '/logs_-1/lamb0.0/noise0.0/delay0/neurons64/inst'
                   + str(i) + '/9afbb8777a/accuracies.npz')
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

fig_dir = os.path.join(PATH, 'data_trainedwithnoise')
fig.savefig(os.path.join(fig_dir, 'acc_across_train_inst' + str(INST) +
                         '.svg'))
plt.close()


acc_dpa_dual = []
acc_gng_dual = []
acc_dpa_dpa = []
acc_gng_gng = []
n_epochs = []

for i in range(INST):
    data = np.load(PATH + '/logs_-1/lamb0.0/noise0.0/delay0/neurons64/inst'
                   + str(i) + '/9afbb8777a/accuracies.npz')
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


fig.savefig(os.path.join(fig_dir, 'mean_acc_across_train.svg'))
plt.close()


# Plot accuracy of DPA in dual task against accuracy of DPA in dpa alone
data = np.load(os.path.join(fig_dir, 'data_' + str(gng) + '_' +
                            str(l) + '_' + str(delay) + '_i' + str(INST) +
                            '_n' + str(noise_rng[0]) + '-' +
                            str(noise_rng[-1]) + '_neu' +
                            str(num_neurons[0]) + '-' +
                            str(num_neurons[-1])))

dual_acc = data['acc'][3]
dpa_acc = data['acc'][5]
n = np.arange(INST)

ax = plt.figure()
ax.scatter(dpa_acc, dual_acc)
ax.plot([0, 1], [0, 1], transform=ax.transAxes)

for i, txt in enumerate(n):
    ax.annotate(txt, (dpa_acc[i], dual_acc[i]))









