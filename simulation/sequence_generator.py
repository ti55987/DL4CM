#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 13:58:21 2025

This script creates stimulus sequences with controlled delays.

@author: Prof. Dr. Anne Collins, adapted to Python code by Franziska Usée
"""

# import dependencies
import numpy as np                                                              # numerical operations
import matplotlib.pyplot as plt                                                 # visualization
from scipy.stats import chisquare                                               # probability distributions
import pandas as pd                                                             # dataframe management
import os                                                                       # path management

# note to Ti-Fen: that's just for exporting my generated sequences
# definition of directories
wdir = os.getcwd()                                                              # current working directory
odir = os.path.join(os.sep.join(wdir.split(os.sep)[:-1]), "Experiment", 
                    "Randomization")                                            # output directory

# function definition
def create_stim_sequence(reps, ns, show_plot=False):
    
    """
    This function creates a stimulus sequence with controlled delays.
    
    Inputs:
        reps: number of repetitions (integer)
        ns  : number of stimuli (integer)
    
    Output:
        seq : generated stimulus sequence (list)
   
    """
    # initialization
    seq       = np.tile(np.arange(1, ns+1), reps)                               # start with simple sequence, repeating the same stimulus order [reps] times
    beta      = 5                                                               # softmax parameter
    criterion = False                                                           # criterion for "good" sequence (as indicated by output from Pearson’s chi-squared test)                                   

    # repeat until criterion is met
    while not criterion:
        
        # initialization of delays and counters
        delays = 1 + np.floor(np.ones(2*ns-1) * reps/2).astype(int)             # max. delay: 2 * set size [ns] -1
        count  = (reps * np.ones(ns)).astype(int) + 1                           # counter: numpy array of shape (ns, ), with all entries being equal to number of repetitions [reps] +1
        seq    = list(range(1, ns+1))                                           # start sequence: [1, 2, ..., ns]
        last   = list(range(1, ns+1))                                           # last presentation index of each stimulus

        # generation of a sequence
        for t in range(ns+1, ns*(reps+1)+1):                                    # start filling positions from (ns+1) to (ns*(reps+1)+1)
            Q = np.zeros(ns)                                                    # initialize "urgency for choosing stimulus" metric for each stimulus; numpy array of shape (ns, )
            L = np.zeros(ns, dtype = int)                                       # initialize last presentation metric for each stimulus, numpy array of shape (ns, )
            
            for i in range(ns):                                                 # iterate through stimuli
                idx_delay = t - last[i]-1                                       # index initialization
                idx_delay = np.clip(idx_delay, 0, len(delays)-1)                # clip index values to maximum of delays 
                Q[i]      = delays[idx_delay] + count[i]                        # value update; compute "urgency" to present stimulus i
                L[i]      = t - last[i]                                         # value update; track how long ago stimulus i was last presented

            # decision rule for which stimulus to present next
            #print(L)
            if np.max(L) == delays.shape[0]:                                    # if maximum in L equals maximum delay (i.e., one stimulus with delay == max. delay),  
                choice = np.argmax(L)                                           # just choose the stimulus with longest delay
            else:
                #print(Q)
                softmax = np.exp(beta * Q)                                      # otherwise, compute softmax probabilities 
                softmax = softmax / np.sum(softmax)
                #print(softmax)
                ps      = np.insert(np.cumsum(softmax), 0, 0)                   # insert 0 in numpy array (first position)
                r       = np.random.rand()                                      # uniform random sampling [0,1]
                choice  = np.where(ps < r)[0][-1]                               # select stimulus for which ps < r   
                #print(choice)
 
            # add selected stimulus to sequence
            seq.append(choice+1)

            # update last, delays, count
            last[choice]              = t                                       # update last occurrence of chosen stimulus
            idx_delay_choice          = L[choice]-1                          
            idx_delay_choice          = np.clip(idx_delay_choice, 0, len(delays)-1)
            delays[idx_delay_choice] -= 1                                       # reduce delays count for stimulus choice
            count[choice]            -= 1                                       # reduce remaining stimulus repetitions

        # analyze the sequence
        alldelays = []
        last_seen = np.zeros(ns, dtype = int)
        dseq      = []

        for t_idx, s in enumerate(seq):
            stim_idx = s-1  
            if last_seen[stim_idx] > 0:
                alldelays.append(t_idx+1 - last_seen[stim_idx])                 # how long since the same stimulus appeared
                dseq.append(s)
            last_seen[stim_idx] = t_idx+1

        alldelays = np.array(alldelays)                                         # sequence with all delays
        dseq      = np.array(dseq)

        # compute the delay distribution
        if len(alldelays) > 0:
            max_delay = np.max(alldelays)                                       # maximum delay
            distr     = np.zeros(max_delay+1, dtype = int)                      # initialization of numpy array of shape (max_delay, ) with all entries being 0

            # count frequency of each delay
            for delay_val in alldelays:
                distr[delay_val] += 1
            
            #print(distr)
            distr = distr[1:]                                                   # remove zero index, delays start from 1
            #print(distr)
            # Pearson's chi-squared test
            # H0: observed delay frequencies are obtained by independent sampling 
            # of N observations from a categorical distribution with given expected 
            # frequencies
            expected = np.mean(distr) * np.ones_like(distr)                     # expected distribution: uniform distribution over all delays 
            _, p     = chisquare(f_obs = distr, 
                                 f_exp = expected)

            # visualization
            if show_plot:
                fig       = plt.figure(figsize = (8,5))  
                plt.clf()                                                           # clear current figure
                plt.plot(distr, "o-")                                               # line plot
                plt.xlabel("Delay")                                                 # x-axis label
                plt.xticks(np.arange(0, len(distr)), np.arange(1, len(distr)+1))
                plt.ylabel("Frequency")
                plt.title("Delay distribution")
                plt.pause(0.1)
                
                # plotting style
                fig.tight_layout()

            # check criterion
            criterion = p > 0.05 and ((np.max(distr) - np.min(distr)) < 2)
            
        else:
            criterion = False

    return np.array(seq)-1

# # function application
# for i in range(1,5):
#     set_size = 4
#     seq      = create_stim_sequence(9, set_size)
#     #print(seq)

#     # export generated sequence
#     df       = pd.DataFrame({"seq": seq})                                       # create a dataframe with a column "seq"
#     filename = "test_phase_order_" + str(i) + "_set_size_" + str(set_size) + ".xlsx"
#     df.to_excel(os.path.join(odir, filename), index = False) 