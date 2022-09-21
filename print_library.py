#!/usr/bin/python\

import numpy as np
import matplotlib as plt
# import multiprocess as mp

from common_library import *

def convert_states(states,state_size):
    x = list()
    for index in range(len(states)):
        x.append(getSubstates(states[index][0:2**state_size]))
    x = np.array(x)
    return x

def convert_states_bis(states,state_size):
    x = list()
    for index in range(len(states)):
        s = states[index][0:2**state_size]
        #print(np.unique(s))
        #print(np.sum(np.unique(s)))
        indeces = np.argwhere(s != 0 )
        #print(indeces)
        tmp = np.zeros(state_size)
        for i in range(len(indeces)):
            #print(indeces[i][0])
            probability = s[indeces[i][0]]
            current_state = np.zeros(2**state_size)
            current_state[indeces[i][0]] = 1
            tmp = tmp + getSubstates(current_state)*probability
            
        #print(tmp)
        
        x.append(tmp)
        
    x = np.array(x)
    return x
    
def print_single_row(binary_states):
    # Stack some immage for having a bigger image
    states = np.stack((binary_states,binary_states,binary_states))
    plt.imshow(states, cmap='Greys_r') 
    plt.stem(200,2.5,  markerfmt='')
    plt.stem(200,-0.5,  markerfmt='')
    
    plt.tick_params(left=False)  # remove the ticks

    # plt.gca().axes.get_yaxis().set_visible(False)
    
    return

def print_evolution_state(states, state_size):
    """
     Form the input in state-space of the evolution of the system print an rappresentation 
     of the state of each node in the time, need the figure to be inizialied and showed outside
    """    
    vector_states = convert_states(states,state_size)
    #print(vector_states[:,1].reshape(-1))
    for index in range(1, state_size+1):
        axe = plt.subplot(state_size , 1, index)
        print_single_row(vector_states[:,index-1].reshape(-1))
        axe.set(yticklabels=[])  
        plt.setp(axe, ylabel='g'+str(index))
    plt.xlabel('time')
        
def print_evolution_state_bis(states, state_size):
    """
     Form the input in state-space of the evolution of the system print an rappresentation 
     of the state of each node in the time, need the figure to be inizialied and showed outside
    """    
    print(state_size)
    vector_states = convert_states_bis(states,state_size)
    for index in range(1, state_size+1):
        plt.subplot(state_size , 1, index)
        print_single_row(vector_states[:,index-1].reshape(-1))

    
def main():
    state_size = 12 # get this from the file
    ax = np.load("ax.npy", allow_pickle=True)
    plt.figure(figsize=(15,6))
    print_evolution_state(ax, state_size)
    plt.savefig("jamaica.png")
    
    
    print("------")
    state_size = 12 # get this from the file
    ax = np.load("axe.npy", allow_pickle=True)
    plt.figure(figsize=(15,6))
    print_evolution_state_bis(ax, state_size)
    plt.savefig("jamaica_error.png")
    

if __name__ == "__main__":
    main()

