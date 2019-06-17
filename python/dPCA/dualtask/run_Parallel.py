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
noise_rng = np.array([0.4])
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
        'alr_hps': alr_hps
        }

    # Create DualTask object
    dt = DualTask(**hps)
    # Train the RNN instance for the specific noise
    dt.train()

# Get inputs and outputs from example trials
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

