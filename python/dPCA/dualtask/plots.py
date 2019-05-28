#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 09:18:34 2019

@author: joan
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import data
import data3task
from RecurrentWhisperer import RecurrentWhisperer
from DualTask import DualTask
import random

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
    hps = self.hps
    n_batch = self.hps.data_hps['n_batch']
    n_time = self.hps.data_hps['n_time']
    gng_time = self.hps.data_hps['gng_time']
    n_plot = 6
    FIG_WIDTH = 6  # inches
    FIX_HEIGHT = 9  # inches
    fig = plt.figure(figsize=(FIG_WIDTH, FIX_HEIGHT),
                          tight_layout=True)
#        n_plot = 10
    dpa2_time = data['vec_tau']
    if gng_time == -1:
        task_type = data['task_choice']
    else:
        task_type = 0

    f = plt.figure(fig.number)
    plt.clf()

    inputs = data['inputs']
    output = data['output']
    predictions = dt.predict(data)
    pred_output = predictions['output']
    acc_dpa = predictions['ev_acc_dpa']

    if stop_time is None:
        stop_time = n_time

    time_idx = range(start_time, stop_time)

    for trial_idx in range(n_plot):
        plt.subplot(n_plot, 1, trial_idx+1)
        if n_plot == 1:
            plt.title('Example trial', fontweight='bold')
        else:
            if gng_time == -1:
                plt.title('Example trial %d | Task %d | Acc %d' 
                          % (trial_idx + 1, task_type[trial_idx], acc_dpa),
                          fontweight='bold')
            else:
                plt.title('Example trial %d | Acc %d'
                          % (trial_idx + 1, acc_dpa), fontweight='bold')

        self._plot_single_trial(
            inputs[trial_idx, time_idx, :],
            output[trial_idx, time_idx, :],
            pred_output[trial_idx, time_idx, :])

        # Only plot x-axis ticks and labels on the bottom subplot
        if trial_idx < (n_plot-1):
            plt.xticks([])
        else:
            plt.xlabel('Timestep', fontweight='bold')

    plt.ion()
    plt.show()
    plt.pause(1e-10)
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
     
         

'''Train an RNN with a given noise and compute the value of the accuracy
 of its predictions'''
# Hyperparameters for AdaptiveLearningRate
alr_hps = {'initial_rate': 0.1}
# Noise range for the input to the RNN
noise = np.array([0.0])
#noise_rng = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
# Time of appearence of the go- no go task. 0 for no task. if gng_rng = [-1] 
# then it runs a ramdom trial either of the dualtask, dpa alone or gng alone.
gng = np.array([-1])
#lamb = np.array([0.0])
lamb = np.array([0.0])
delay = np.array([0])
neuron = np.array([64])
# number of RNN instances
inst = 1

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
        'delay_max': delay},
    'alr_hps': alr_hps
    }

# Create DualTask object
dt = DualTask(**hps)
# Train the RNN instance for the specific noise
dt.train()

# Get inputs and outputs from example trials
random.seed(noise*inst)
example_trials = dt.generate_dualtask_trials()

is_lstm = dt.hps.rnn_type == 'lstm'