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
PATH = '/home/joan/dPCA/python/dPCA'
sys.path.insert(0, PATH) 
from DualTask import DualTask
from dPCA import dPCA
from joblib import Parallel, delayed
import multiprocessing

# *****************************************************************************
# STEP 1: Train RNNs to solve the dual task *********************************
# *****************************************************************************
# Noise range for the input to the RNN
noise_rng = np.array([0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6])
# Time of appearence of the go- no go task. 0 for no task.
gng_rng = np.array([0, 10])
lamb = 0, delay_max = 0


def trainDualTask(noise, gng, inst):
    '''Train an RNN with a given noise and compute the value of the accuracy
    of its predictions'''
    # Hyperparameters for AdaptiveLearningRate
    alr_hps = {'initial_rate': 0.1}

    # Hyperparameters for DualTask
    # See DualTask.py for detailed descriptions.
    hps = {
        'rnn_type': 'vanilla',
        'n_hidden': 256,
        'min_loss': 1e-6,  # 1e-4
        'min_learning_rate': 1e-5,
        'max_n_epochs': 5000,
        'do_restart_run': True,
        'log_dir': './logs_' + str(gng) + '/' + str(inst),
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
    return ([acc_dpa, acc_gng])


# Train various RNNs with diferent noise
for gng in gng_rng:

    datalist = []
    f = plt.figure()
    plt.clf()
    # number of RNN instances
    INST = 10

    for noise in noise_rng:

        numcores = multiprocessing.cpu_count()
        acc = Parallel(n_jobs=numcores)(delayed(trainDualTask)(
                noise, gng, inst) for inst in range(INST))
        acc_dpa = []
        acc_gng = []
        for i in range(INST):
            acc_dpa.append(acc[i][0])
            acc_gng.append(acc[i][1])
        # Save data in a list
        NOISE = np.repeat(noise, INST)
        datalist.append([NOISE, acc_dpa, acc_gng])
        # Plot loss / accuracy for the different noise- instances
        plt.figure(f.number)
#        plt.plot(noise, loss_dpa, '+')
#        plt.plot(noise, loss_gng, 'v')
        plt.plot(NOISE, acc_dpa, '+', color='k')
        plt.plot(NOISE, acc_gng, 'v', color='k')
        plt.xlabel('Noise')
        plt.ylabel('Accuracy')
        plt.ion()
        plt.draw()
        plt.show()
        plt.pause(0.01)

    # save data and figure
    data = {'datalist': datalist}

    fig_dir = os.path.join(PATH, 'data_trainedwithnoise')
    try:
        os.mkdir(fig_dir)
    except OSError:
        np.savez(os.path.join(fig_dir, 'data_' + str(gng) + '_'
                              + str(lamb) + '_' + str(delay_max)
                              + '_i' + str(INST) + '_n' + str(noise_rng[0])
                              + '-' + str(noise_rng[-1])), **data)
        plt.savefig(os.path.join(fig_dir, 'acc_noise_' + str(gng) + '_'
                                 + str(lamb) + '_' + str(delay_max)
                                 + '_i' + str(INST) + '_n' + str(noise_rng[0])
                                 + '-' + str(noise_rng[-1]) + '.png'))
    else:
        np.savez(os.path.join(fig_dir, 'data_' + str(gng) + '_'
                              + str(lamb) + '_' + str(delay_max)
                              + '_i' + str(INST) + '_n' + str(noise_rng[0])
                              + '-' + str(noise_rng[-1])), **data)
        plt.savefig(os.path.join(fig_dir, 'acc_noise_' + str(gng) + '_'
                                 + str(lamb) + '_' + str(delay_max)
                                 + '_i' + str(INST) + '_n' + str(noise_rng[0])
                                 + '-' + str(noise_rng[-1]) + '.png'))

'''Plot data together'''
# Loading the data for task without distractor
data = np.load(fig_dir + '/data_0_0_0' + '_i' + str(INST) + '_n' +
               str(noise_rng[0]) + '-' + str(noise_rng[-1]) + '.npz')
data = np.hstack(data['datalist'][i] for i in range(noise_rng.shape[0]))

# Compute the mean accuracy across instances
mean_acc = []
std = []
for n in noise_rng:
    mean_acc.append(np.mean(data[1, data[0, :] == n]))
    std.append(np.std(data[1, data[0, :] == n]))
# Plot with error bars of the accuracy / loss
plt.figure()
plt.errorbar(noise_rng, mean_acc, yerr=std, marker='+',
             label='DPA accuracy dpa gng0')

# Loading data for task with distractor
data = np.load(fig_dir + '/data_10_0_0' + '_i' + str(INST) + '_n' +
               str(noise_rng[0]) + '-' + str(noise_rng[-1]) + '.npz')
data = np.hstack(data['datalist'][i] for i in range(noise_rng.shape[0]))

# Compute the mean accuracy across instances
mean_acc = []
std = []
for n in noise_rng:
    mean_acc.append(np.mean(data[1, data[0, :] == n]))
    std.append(np.std(data[1, data[0, :] == n]))

# Plot with error bars of the accuracy / loss
plt.errorbar(noise_rng, mean_acc, yerr=std, marker='+',
             label='DPA accuracy gng10')


plt.xlabel('Noise')
plt.ylabel('Mean accuracy')
plt.legend()
plt.show()