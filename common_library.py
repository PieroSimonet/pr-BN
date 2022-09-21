import numpy as np
from math import floor, log
import matplotlib.pyplot as plt


def checkState(state):
    return 0

def getSubstates(state):
    """
    Get each node state (1=up, 0=down) given the state of the system
    """
    x = np.zeros(int(log(state.shape[0], 2)))
    
    size = state.shape[0]
    
    pos = np.argwhere(state)[0][0]
        
    for index in range(x.shape[0]):
        if (pos < size/2):
            x[index] = 1
        else:
            pos -= size/2
            
        size /= 2
    
    return x

def getState(substate):
    """
    Get the state of the system given an array with the state of each node
    """
    x = np.zeros(2**substate.shape[0])
    
    pos = 0
    for index in range(len(substate)):
        if (substate[index] == 0):
            pos += 2**(len(substate)-index-1)
            
    x[int(pos)] = 1
    return x

def printState(state):
    sub = getSubstates(state)
    for index in range(len(sub)):
        print("x"+str(index+1)+" ")
        if (sub[index]==1):
            print("active\n")
        else:
            print("not active\n")

def getXD(start,final):
    """
    Given two states of the system, give back the indeces of the nodes where they differ
    """
    sub_start = getSubstates(start)
    sub_final = getSubstates(final)
    
    XD = np.argwhere(sub_start!=sub_final).reshape(-1)

    return XD

def getRandState(dim):
    size = 2**dim
    x = np.zeros((size,))
    x[np.random.randint(size)]=1
    return x

def getAlpha(x):
    return sum(x)

def getEstimation(xo,xe):
    x = np.multiply(xo,xe)
    alpha = getAlpha(x)
    x /= alpha
    return x, alpha
    
def set_damage(state, node, damage_type, cascade_update=False):
    """
    The stuck at fault -> very tipical of cancer, needed for simulations
    """
    sub_state = getSubstates(state)
    if (sub_state[node] != damage_type ):   
        sub_state[node] = damage_type
        state = getState(sub_state)
        if (cascade_update):
            state = np.concatenate((state,np.array([node])))
    return state

def semi_tensor(A,B):
    #print(A)
    #print(B)
    if (len(A.shape)==1):
        n=len(A)
    else:
        n = A.shape[1]
    p = B.shape[0]
    a = np.lcm(n,p)
    d = np.kron(A, np.eye(int(a/n)))
    #print(d)
    s = np.kron(B, np.eye(int(a/p)))
    #print(s)
    return np.matmul(d,s)
    
def get_most_probabily_state(state):
    max = np.argmax(state)
    new_state = np.zeros(len(state))
    new_state[max] = 1
    return new_state

def get_all_substate(state, len_state):
    x = []
    p = []
    for i in range(2**len_state):
        if (state[i]>0):
            tmp = np.zeros(2**len_state)
            tmp[i] = 1
            x.append(tmp)
            p.append(state[i])
    return x,p
    
def state2graph(states,size):
    x = []
    size = 2**size
    for state in states:
        tmp = 0
        for p in range(size):
            tmp += state[size-1-p]*(p+1)
        x.append(tmp)
    x = np.array(x)
    plt.plot(x)
