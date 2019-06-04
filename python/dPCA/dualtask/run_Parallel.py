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
from matplotlib import cm
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition

# *****************************************************************************
# STEP 1: Train RNNs to solve the dual task *********************************
# *****************************************************************************
# Noise range for the input to the RNN
noise_rng = np.array([0.2])
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
    pred_out = example_predictions['output']
    inputs = example_trials['inputs']
    outputs = example_trials['output']
    stim_conf = example_trials['stim_conf']
    vec_tau = example_trials['vec_tau']
    vec_acc_dual = example_predictions['vec_acc_dpa_dual']
    vec_acc_dpa = example_predictions['vec_acc_dpa_dpa']
    if gng == -1:
        task_type = example_trials['task_choice']
    else:
        task_type = 0
    stim_conf = example_trials['stim_conf']


    return [acc_dpa, acc_gng, state, task_type, acc_dpa_dual, acc_gng_dual,
            acc_dpa_dpa, acc_gng_gng, vec_acc_dual, vec_acc_dpa,
            pred_out, inputs, outputs, stim_conf, vec_tau]

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
                    inputs = []
                    output = []
                    pred_out = []
                    stim_conf = []
                    vec_tau = []
                    vec_acc_dual = []
                    vec_acc_dpa = []
                    stim_conf = []
                    for i in range(INST):
                        acc_dpa.append(ops[i][0])
                        acc_gng.append(ops[i][1])
                        state.append([noise, ops[i][2]])
                        task.append(ops[i][3])
                        acc_dpa_dual.append(ops[i][4])
                        acc_gng_dual.append(ops[i][5])
                        acc_dpa_dpa.append(ops[i][6])
                        acc_gng_gng.append(ops[i][7])
                        vec_acc_dual.append(ops[i][8])
                        vec_acc_dpa.append(ops[i][9])
                        pred_out.append(ops[i][10])
                        output.append(ops[i][12])
                        inputs.append(ops[i][11])
                        stim_conf.append(ops[i][13])
                        vec_tau.append(ops[i][14])
                        
                    acc.append([noise, acc_dpa, acc_gng, acc_dpa_dual,
                                acc_gng_dual, acc_dpa_dpa, acc_gng_gng,
                                vec_acc_dual, vec_acc_dpa])
                  
    
        # save data and figure
            data = {'acc': acc, 'state': state, 'task': task, 'inputs': inputs,
                    'output': output, 'pred_out': pred_out,
                    'stim_conf': stim_conf, 'vec_tau': vec_tau}
        
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

# Runs DPA + GNG task
else:
    # Train various RNNs with diferent noise
    for gng in gng_rng:
        
        for l in  lamb:
            acc = []
            state = []
            
            for delay in delay_max:
        
                for noise in noise_rng:
            
                    for neuron in num_neurons:
                        numcores = multiprocessing.cpu_count()
                        ops = Parallel(n_jobs=numcores)(delayed(
                                trainDualTask)(noise, gng, inst, l, delay,
                                             neuron) for inst in range(INST))
                        
                        # Save data in a list
                        NOISE = np.repeat(l, INST)
                        acc_dpa = []
                        acc_gng = []
                        task_type = []
                        for i in range(INST):
                            acc_dpa.append(ops[i][0])
                            acc_gng.append(ops[i][1])
                            state.append([noise, ops[i][2]])
                                    
                        acc.append([noise, acc_dpa, acc_gng])
        
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
                else:
                    np.savez(os.path.join(fig_dir, 'data_' + str(gng) + '_'
                                          + str(l) + '_' + str(delay)
                                          + '_i' + str(INST) + '_n' + str(noise_rng[0])
                                          + '-' + str(noise_rng[-1])
                                          + '_neu' + str(num_neurons[0])
                                          + '-' + str(num_neurons[-1])), **data)

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
noise = noise_rng[0]
plt.figure()
label_added = False
for i in range(INST):
    data = np.load(PATH + '/logs_-1/lamb0.0/noise' + str(noise) + '/delay0/neurons64/inst'
                   + str(i) + '/cdd99a83bc/accuracies.npz')
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
                         '_noise_' + str(noise) + '.png'))
plt.close()


#Plot for the mean accuracy of DPA and dual with all trials

acc_dpa_dual = []
acc_gng_dual = []
acc_dpa_dpa = []
acc_gng_gng = []
n_epochs = []

for i in range(INST):
    data = np.load(PATH + '/logs_-1/lamb0.0/noise' + str(noise) + '/delay0/neurons64/inst'
                   + str(i) + '/cdd99a83bc/accuracies.npz')
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


fig.savefig(os.path.join(fig_dir, 'mean_acc_across_train' + str(noise) + '.png'))
plt.close()



# Plot accuracy of DPA in dual task against accuracy of DPA in dpa alone with numbers
data = np.load(os.path.join(fig_dir, 'data_-1_0.0_0_i50_n' + str(noise) + '-' +
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
f.savefig(os.path.join(fig_dir, 'dpa_vs_dual_accnumber_' + str(noise) + '.png'))

# Plot accuracy of DPA in dual task against accuracy of DPA in dpa alone without numbers
data = np.load(os.path.join(fig_dir, 'data_-1_0.0_0_i50_n' + str(noise) + '-' +
                            str(noise) + '_neu64-64.npz'))

dual_acc = data['acc'][0][3]
dpa_acc = data['acc'][0][5]
n = np.arange(INST)

f, ax = plt.subplots()
ax.scatter(dpa_acc, dual_acc, color='b', s=5)
ax.plot([0.4, 1], [0.4, 1], ls='--', color='grey')
plt.xlabel('DPA acc')
plt.ylabel('dual DPA acc')
f.savefig(os.path.join(fig_dir, 'dpa_vs_dual_acc_' + str(noise) + '.png'))


# Bar plot acc dual vs acc dpa

plt.figure()
x = np.arange(2)
means = [np.mean(dual_acc), np.mean(dpa_acc)]
plt.bar(x, means, color='b', width=0.3)
plt.xticks(x, ('Dual-task', 'DPA task'))
plt.title('Mean accuracy')
plt.savefig(os.path.join(fig_dir, 'mean_acc_bar' + str(noise) + '.png'))
plt.close('all')

## Plot accuracy across training of the instances where acc DPA > acc DPA 
## dual at the end of the training and also acc dpa>0.55
#
#acc_dpa_dual = []
#acc_dpa_dpa = []
#n_epochs = []
#inst_cond = []
#for i in range(INST):
#    data = np.load(PATH + '/logs_-1/lamb0.0/noise' + str(noise) + '/delay0/neurons64/inst'
#                   + str(i) + '/9afbb8777a/accuracies.npz')
#    if dpa_acc[i] > 0.55:
#        if dual_acc[i] < dpa_acc[i]:
#            acc = data['acc_dpa_dual']
#            acc_dpa_dual.append(acc)
#            acc = data['acc_dpa_dpa']
#            acc_dpa_dpa.append(acc)
#            n = data['n_epochs']
#            n_epochs.append(n)
#            inst_cond.append(i)
#            
##inst_cond = np.array(inst_cond)
#n = np.shape(n_epochs)[0]
#min_epochs = np.min(tuple(n_epochs[i] for i in range(n)))//10
#acc_dpa_dualstack = acc_dpa_dual[0][0:min_epochs]
#acc_dpa_dpastack = acc_dpa_dpa[0][0:min_epochs]
#for i in range(n-1):
#    acc_dpa_dualstack = np.column_stack((acc_dpa_dualstack,
#                                         acc_dpa_dual[i+1][0:min_epochs]))
#    acc_dpa_dpastack = np.column_stack((acc_dpa_dpastack,
#                                        acc_dpa_dpa[i+1][0:min_epochs]))
#
#acc_dpa_dualmean = np.mean(acc_dpa_dualstack, axis=1)
#acc_dpa_dpamean = np.mean(acc_dpa_dpastack, axis=1)
#
#acc_dpa_dualstd = np.std(acc_dpa_dualstack, axis=1)
#acc_dpa_dpastd = np.std(acc_dpa_dpastack, axis=1)
#
#epochs = np.arange(min_epochs)
#
#plt.figure()
#plt.plot(epochs, acc_dpa_dualmean, label='Dual DPA', color='r')
#plt.plot(epochs, acc_dpa_dpamean, label='DPA DPA', color='g')
#
#for i, num in enumerate(inst_cond):
#    plt.plot(epochs, acc_dpa_dualstack[:, i], color='r', alpha=0.2)
#    plt.plot(epochs, acc_dpa_dpastack[:, i], color='g', alpha=0.2)
#    plt.annotate(num, (epochs[-1] + 1, acc_dpa_dualstack[-1, i]))
##plt.fill_between(epochs, acc_dpa_dualmean-acc_dpa_dualstd, acc_dpa_dualmean+
##                 acc_dpa_dualstd, color='r', alpha=0.5)
##plt.fill_between(epochs, acc_dpa_dpamean-acc_dpa_dpastd, acc_dpa_dpamean+
##                 acc_dpa_dpastd, color='g', alpha=0.5)
#
##plt.xlim([0, 100])
#plt.legend()
#plt.xlabel('Epoch')
#plt.ylabel('Mean accuracy')
#fig = plt.gcf()
#plt.show()
#
#
#fig.savefig(os.path.join(fig_dir, 'mean_acc_across_train_dual_vs_dpa' + str(noise) + '.svg'))
#plt.close()



# Count which number of stimulus pairs (s1-s3/s4 or s2-s3/s4) are correct
# for the conditions that appears s5 or s6 during the distractor

fig_dir_dir = os.path.join(PATH, 'plots')
data = np.load(os.path.join(fig_dir, 'data_-1_0.0_0_i50_n' + str(noise) + '-' +
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
    
    
       
    
    if not os.path.exists(fig_dir_dir):
        os.mkdir(fig_dir_dir)
        plt.savefig(os.path.join(fig_dir_dir, 'Inst' + str(i) + '.png'))
    else:
        plt.savefig(os.path.join(fig_dir_dir, 'Inst' + str(i) + '.png'))
    plt.close()
    
    matdual_inst.append(matdual)
    matdpa_inst.append(matdpa)

    

