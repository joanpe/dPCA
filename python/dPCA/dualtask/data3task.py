#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 15:05:34 2019

@author: molano
"""
import numpy as np
import matplotlib.pylab as plt
import random

def get_inputs_outputs(n_batch, n_time, n_bits, gng_time, lamb,
                       delay_max, noise, mat_conv=[0, 1]):
    gng_time = 10
    # Get random task type trials
    task_choice = np.random.choice(3, size=n_batch)
    inputs = []
    outputs =[]
    stim_conf = []
    vec_tau = []
    for ind_btch in range(n_batch):
        if task_choice[ind_btch]==0:
            # Get trial for Dualtask
            inp, out, conf, tau = dual_task(n_time, n_bits, gng_time,
                                            lamb, delay_max, noise,
                                            mat_conv=[0, 1])
        elif task_choice[ind_btch]==1:
            # Get trial for DPA
            inp, out, conf, tau = dpa(n_time, n_bits, gng_time, lamb,
                                      delay_max, noise, mat_conv=[0, 1])
        else:
            # Get trial for GNG
            inp, out, conf, tau = gng(n_time, n_bits, gng_time, lamb,
                                      delay_max, noise, mat_conv=[0, 1])

        # Add together all trials in a batch
        inputs.append(inp)
        outputs.append(out)
        stim_conf.append(conf)
        vec_tau.append(tau)
        
    inputs = np.array(inputs)
    outputs = np.array(outputs)
    stim_conf = np.array(stim_conf)
    stim_conf.reshape(n_batch, 4)
    vec_tau = np.array(vec_tau)
    vec_tau.reshape(n_batch, 1)

    return {'inputs': inputs, 'output': outputs, 'task_choice': task_choice,
            'stim_conf': stim_conf, 'vec_tau': vec_tau}

# Dual Task stimulus structure
def dual_task(n_time, n_bits, gng_time, lamb, delay_max, noise,
              mat_conv=[0, 1]):
    # inputs mat
    inputs = np.zeros([n_time, n_bits])
    # build dpa structure
    dpa_stim1 = np.arange((n_bits-2)/2)
    stim1_seq, choice1 = get_stims(dpa_stim1, 1)
    dpa_stim2 = np.arange((n_bits-2)/2, (n_bits-2))
    stim2_seq, choice2 = get_stims(dpa_stim2, 1)
    # ground truth dpa
    gt_dpa = choice1 == choice2
    gt_dpa = gt_dpa*1

    # build go-noGo task:
    gng_stim = np.arange((n_bits-2), n_bits)
    gng_stim_seq, gt_gng = get_stims(gng_stim, 1)

    # DPA1 stimulus
    inputs[1, stim1_seq] = 1
    # dpa2 presented at delay gng_time + random delay between gng_time + 2
    # and gng_time + 2 + delay_max. tau in range[0,9]
    if delay_max == 0:
        inputs[n_time-5, stim2_seq] = 1
        tau = 0
    else:
        # tau= time at which dpa2 appears
        tau = np.random.choice(delay_max, size=1)+gng_time+2
        if tau < n_time:
            inputs[tau, stim2_seq] = 1
        else:
            raise ValueError('Delay exceed trial time.')

    # Parametrization of gng stimulus
    inputs[gng_time-1, gng_stim_seq] = 1-lamb
    # Example: S5 --> index 4, S1 --> index 0, mat_conv[S5] = 0
    inputs[gng_time-1, mat_conv[gt_gng]] = lamb

    # output (note that n_bits could actually be 1 here because we just
    # need one decision. I kept it as it is for the flipFlop task
    # to avoid issues in other parts of the algorithm)
    outputs = np.zeros([n_time, n_bits])
    outputs[n_time-1, 0] = gt_dpa
    
    # distractor time = gng_time
    outputs[gng_time, 0] = gt_gng

    # Adding noise to the inputs
    inputs += np.random.normal(scale=noise, size=inputs.shape)

    # stim configuration
    stim_conf = np.array([choice1, choice2, gt_gng,
                                gt_dpa])

    return inputs, outputs, stim_conf, tau

# DPA task stimulus structure
def dpa(n_time, n_bits, gng_time, lamb, delay_max, noise,
        mat_conv=[0, 1]):
    gt_gng = 2
    # inputs mat
    inputs = np.zeros([n_time, n_bits])
    # build dpa structure
    dpa_stim1 = np.arange((n_bits-2)/2)
    stim1_seq, choice1 = get_stims(dpa_stim1, 1)
    dpa_stim2 = np.arange((n_bits-2)/2, (n_bits-2))
    stim2_seq, choice2 = get_stims(dpa_stim2, 1)
    # ground truth dpa
    gt_dpa = choice1 == choice2
    gt_dpa = gt_dpa*1

    # DPA1 stimulus
    inputs[1, stim1_seq] = 1
    # dpa2 presented at delay gng_time + random delay between gng_time + 2
    # and gng_time + 2 + delay_max. tau in range[0,9]
    if delay_max == 0:
        inputs[n_time-5, stim2_seq] = 1
        tau = 0
    else:
        # tau= time at which dpa2 appears
        tau = np.random.choice(delay_max, size=1)+gng_time+2
        if tau < n_time:
            inputs[tau, stim2_seq] = 1
        else:
            raise ValueError('Delay exceed trial time.')

    # output (note that n_bits could actually be 1 here because we just
    # need one decision. I kept it as it is for the flipFlop task
    # to avoid issues in other parts of the algorithm)
    outputs = np.zeros([n_time, n_bits])
    outputs[n_time-1, 0] = gt_dpa

    # Adding noise to the inputs
    inputs += np.random.normal(scale=noise, size=inputs.shape)

    # stim configuration
    stim_conf = np.array([choice1, choice2, gt_gng,
                                gt_dpa])

    return inputs, outputs, stim_conf, tau

# Go no Go task stimulus structure
def gng(n_time, n_bits, gng_time, lamb, delay_max, noise,
        mat_conv=[0, 1]):
    gt_dpa = 2
    choice1, choice2 = 2, 2
    # inputs mat
    inputs = np.zeros([n_time, n_bits])

    # build go-noGo task
    gng_stim = np.arange((n_bits-2), n_bits)
    gng_stim_seq, gt_gng = get_stims(gng_stim, 1)

    # Parametrization of gng stimulus
    inputs[gng_time-1, gng_stim_seq] = 1-lamb
    # Example: S5 --> index 4, S1 --> index 0, mat_conv[S5] = 0
    inputs[gng_time-1, mat_conv[gt_gng]] = lamb

    # output (note that n_bits could actually be 1 here because we just
    # need one decision. I kept it as it is for the flipFlop task
    # to avoid issues in other parts of the algorithm)
    outputs = np.zeros([n_time, n_bits])
    # distractor time = gng_time
    outputs[gng_time, 0] = gt_gng

    # Adding noise to the inputs
    inputs += np.random.normal(scale=noise, size=inputs.shape)
    
    # Vector with the delays of dp2
    tau = 0

    # stim configuration
    stim_conf = np.array([choice1, choice2, gt_gng,
                                gt_dpa])

    return inputs, outputs, stim_conf, tau

def get_stims(stim, n_batch):
    choice = np.random.choice(stim.shape[0])
#    choice = np.concatenate((np.zeros(int(n_batch/2,)),
#                             np.ones(int(n_batch/2,)))).astype(int)
    stim_seq = stim[choice].astype(int)
    return stim_seq, choice


if __name__ == '__main__':
    plt.close('all')
    n_batch = 10
    n_time = 8
    n_bits = 6
    gng_time = 3
    example_trials = get_inputs_outputs(n_batch, n_time, n_bits, gng_time)
#    print(example_trials['inputs'][0, :, :].T)
#    print('----')
#    print(example_trials['inputs'][1, :, :].T)
#    print('----')
#    print(example_trials['inputs'][2, :, :].T)
#    print('----')
#    print(example_trials['inputs'][3, :, :].T)
#    print('----')
#    print(example_trials['output'].shape)
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.imshow(np.squeeze(example_trials['inputs'][0, :, :].T), aspect='auto')
    plt.subplot(2, 1, 2)
    plt.plot(np.squeeze(example_trials['output'][0, :, 0]))
