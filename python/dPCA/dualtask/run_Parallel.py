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
noise_rng = np.array([0])
#noise_rng = np.array([0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6])
# Time of appearence of the go- no go task. 0 for no task.
gng_rng = np.array([0, 10])
lamb = np.array([0.0])
delay_max = np.array([0])
num_neurons = np.array([4, 8, 16, 32, 64, 128])
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
        str(noise) + '/delay' + str(delay) + '/inst' + str(inst) + '/neurons' +
        str(neuron),
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

    # Adding noise to the inputs (done in data.py)
#        example_trials['inputs'] += np.random.normal(scale=noise,
#                                                     size=example_trials[
#                                                             'inputs'].shape)
    # Compute RNN predictions from example trials
    example_predictions = dt.predict(example_trials,
                                     do_predict_full_LSTM_state=is_lstm)

    # Accuracy of the predictions
    acc_dpa = example_predictions['ev_acc_dpa']
    acc_gng = example_predictions['ev_acc_gng']
    state = example_predictions['state']
    return [acc_dpa, acc_gng, state]


# Train various RNNs with diferent noise
for gng in gng_rng:

    acc = []
    state = []
    f = plt.figure()
    plt.clf()
    
    for l in  lamb:
        
        for delay in delay_max:

            for noise in noise_rng:
        
                for neuron in num_neurons:
                    numcores = multiprocessing.cpu_count()
                    ops = Parallel(n_jobs=numcores)(delayed(
                            trainDualTask)(noise, gng, inst, l, delay,
                                         neuron) for inst in range(INST))
                    
                    # Save data in a list
                    NOISE = np.repeat(neuron, INST)
                    acc_dpa = []
                    acc_gng = []
                    for i in range(INST):
                        acc_dpa.append(ops[i][0])
                        acc_gng.append(ops[i][1])
                        state.append([noise, ops[i][2]])
                    acc.append([noise, acc_dpa, acc_gng])
                    # Plot loss / accuracy for the different noise- instances
                    plt.figure(f.number)
            #        plt.plot(noise, loss_dpa, '+')
            #        plt.plot(noise, loss_gng, 'v')
                    plt.plot(NOISE, acc_dpa, '+', color='k')
                    plt.plot(NOISE, acc_gng, 'v', color='k')
                    plt.xlabel('Num neurons')
                    plt.ylabel('Accuracy')
                    plt.ion()
                    plt.draw()
                    plt.show()
                    plt.pause(0.01)
    
        # save data and figure
            data = {'acc': acc, 'state': state}
        
            fig_dir = os.path.join(PATH, 'data_trainedwithnoise')
            if os.path.isdir(fig_dir) is False:
                os.mkdir(fig_dir)
                np.savez(os.path.join(fig_dir, 'data_' + str(gng) + '_'
                                      + str(l) + '_' + str(delay)
                                      + '_i' + str(INST) + '_n' + str(noise_rng[0])
                                      + '-' + str(noise_rng[-1])
                                      + '_neu' + str(num_neurons[0])
                                      + '-' + str(num_neurons[-1])), **data)
                plt.savefig(os.path.join(fig_dir, 'acc_noise_' + str(gng) + '_'
                                         + str(l) + '_' + str(delay)
                                         + '_i' + str(INST) + '_n' + str(noise_rng[0])
                                         + '-' + str(noise_rng[-1])
                                         + '_neu' + str(num_neurons[0])
                                         + '-' + str(num_neurons[-1]) + '.png'))
            else:
                np.savez(os.path.join(fig_dir, 'data_' + str(gng) + '_'
                                      + str(l) + '_' + str(delay)
                                      + '_i' + str(INST) + '_n' + str(noise_rng[0])
                                      + '-' + str(noise_rng[-1])
                                      + '_neu' + str(num_neurons[0])
                                      + '-' + str(num_neurons[-1])), **data)
                plt.savefig(os.path.join(fig_dir, 'acc_noise_' + str(gng) + '_'
                                         + str(l) + '_' + str(delay)
                                         + '_i' + str(INST) + '_n' + str(noise_rng[0])
                                         + '-' + str(noise_rng[-1])
                                         + '_neu' + str(num_neurons[0])
                                         + '-' + str(num_neurons[-1]) + '.png'))


fig_dir = os.path.join(PATH, 'data_trainedwithnoise')

'''Plot mean data together'''
# Plots for number of neurons against accuracy
# Loading the data for task without distractor
plt.figure()

for l in lamb:
    for delay in delay_max:
        data = np.load(fig_dir + '/data_0_' + str(l) + '_' + str(delay) +
                       '_i' + str(INST) + '_n' + str(noise_rng[0]) + '-' +
                       str(noise_rng[-1]) + '_neu' + str(num_neurons[0])
                       + '-' + str(num_neurons[-1]) + '.npz')
        data = data['acc']
        
        # Compute the mean accuracy across instances
        mean_acc = []
        std = []
        for n in range(num_neurons.shape[0]):
            mean_acc.append(np.mean(data[n][1]))
            std.append(np.std(data[n][1]))
            plt.scatter(np.repeat(num_neurons[n], INST), data[n][1], marker='.', color='b')
       
        # Plot with error bars of the accuracy / loss
        plt.plot(num_neurons, mean_acc, marker='+', ms=15, color='b',
                     label='DPA accuracy dpa gng0 lamb' + str(l))

# Loading data for task with distractor
for l in lamb:
    for delay in delay_max:
        data10 = np.load(fig_dir + '/data_10_' + str(l) + '_' + str(delay) +
                         '_i' + str(INST) + '_n' + str(noise_rng[0]) + '-' +
                         str(noise_rng[-1]) + '_neu' + str(num_neurons[0])
                         + '-' + str(num_neurons[-1]) + '.npz')
        data10 = data10['acc']

        # Compute the mean accuracy across instances
        mean_acc = []
        std = []
        for n in range(num_neurons.shape[0]):
            mean_acc.append(np.mean(data10[n][1]))
            std.append(np.std(data10[n][1]))
            plt.scatter(np.repeat(num_neurons[n], INST), data10[n][1], marker='.', color='r')
            
        # Plot with error bars of the accuracy / loss
        plt.plot(num_neurons, mean_acc, marker='+', ms=15, color='r',
                     label='DPA accuracy gng10 lamb' + str(l))

plt.xlabel('Number of neurons')
plt.ylabel('Mean accuracy')
plt.legend()
plt.show()

if os.path.isdir(fig_dir) is False:
    os.mkdir(fig_dir)
    plt.savefig(os.path.join(fig_dir, 'mean_acc_neurons_'
                             + str(l) + '_' + str(delay)
                             + '_i' + str(INST) + '_n' + str(noise_rng[0])
                             + '-' + str(noise_rng[-1]) + '.png'))
else:
    plt.savefig(os.path.join(fig_dir, 'mean_acc_neurons_'
                             + str(l) + '_' + str(delay)
                             + '_i' + str(INST) + '_n' + str(noise_rng[0])
                             + '-' + str(noise_rng[-1]) + '.png'))

#
## Plots for noise against accuracy
## Loading the data for task without distractor
#for l in lamb:
#    for delay in delay_max:
#        data = np.load(fig_dir + '/data_0_' + str(l) + '_' + str(delay) +
#                       '_i' + str(INST) + '_n' + str(noise_rng[0]) + '-' +
#                       str(noise_rng[-1]) + '.npz')
#        data = data['acc']
#        
#        # Compute the mean accuracy across instances
#        mean_acc = []
#        std = []
#        for n in range(noise_rng.shape[0]):
#            mean_acc.append(np.mean(data[n][1]))
#            std.append(np.std(data[n][1]))
#        # Plot with error bars of the accuracy / loss
#        plt.errorbar(noise_rng, mean_acc, yerr=std, marker='+',
#                     label='DPA accuracy dpa gng0 lamb' + str(l))
#
## Loading data for task with distractor
#for l in lamb:
#    for delay in delay_max:
#        data10 = np.load(fig_dir + '/data_10_' + str(l) + '_' + str(delay) +
#                         '_i' + str(INST) + '_n' + str(noise_rng[0]) + '-' +
#                         str(noise_rng[-1]) + '.npz')
#        data10 = data10['acc']
#
#        # Compute the mean accuracy across instances
#        mean_acc = []
#        std = []
#        for n in range(noise_rng.shape[0]):
#            mean_acc.append(np.mean(data10[n][1]))
#            std.append(np.std(data10[n][1]))
#
#        # Plot with error bars of the accuracy / loss
#        plt.errorbar(noise_rng, mean_acc, yerr=std, marker='+',
#                     label='DPA accuracy dpa gng10' + str(l))
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
#                             + '-' + str(noise_rng[-1]) + '.png'))
#else:
#    plt.savefig(os.path.join(fig_dir, 'mean_acc_noise_'
#                             + str(l) + '_' + str(delay)
#                             + '_i' + str(INST) + '_n' + str(noise_rng[0])
#                             + '-' + str(noise_rng[-1]) + '.png'))


