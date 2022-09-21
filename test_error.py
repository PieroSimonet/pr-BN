import numpy as np
import matplotlib.pyplot as plt
from common_library import getSubstates
from  math import log

def calulate_STN(data):
    m = np.mean(data)
    s = np.std(data)
    if (s==0):
        return 0
    return m/s

    
folder = "data/2022-05-27_12/"

dim = 2**12

print("Load data")
# Normal data
ax  = np.load(folder+"ax.npy", allow_pickle=True)
# Filter data
axe = np.load(folder+"axe.npy", allow_pickle=True)

print("Calculation error vector")
ae = list()
# Since I use the fully observable state I can simple use ax and axe directly
for index in range(len(ax)):
    a = getSubstates(ax[index][0:dim])
    
    s = axe[index][0:dim]
    
    indeces = np.argwhere(s != 0 )
    #print(indeces)
    abs = 0
    # print(len(indeces))
    for i in range(len(indeces)):
        #print(i)
        #print(indeces[i][0])
        probability = s[indeces[i][0]]
        
        #print(probability)

        current_state = np.zeros(dim)
        current_state[indeces[i][0]] = 1
        b = getSubstates(current_state)
        abs = abs + np.count_nonzero(a!=b)*probability
    
    #b = getSubstates([i][0:dim])
    #abs = np.count_nonzero(a!=b) #np.abs(uno-due)
    #print(abs)
    ae.append(abs)

ae = np.array(ae)

time_size = len(ae)
print("Time size    :"+str(time_size))
time_windows = 30 # time_size/10
print("Time windows :"+str(time_windows))

# Calculate the mod

x = list()
stn = list()
for index in range(int(time_size-time_windows)):
    #tmp = np.sum(abs_ae[index:int(index+time_windows)]) ** 2
    a = calulate_STN(ae[index:int(index+time_windows)])
    stn.append(20*log(a))
    #x.append(tmp)

stn = np.array(stn)

plt.figure()
plt.plot(stn)
plt.savefig("stn")

"""
tmp = list()
for index in range(len(x)-int(time_windows)):
    tmp.append(x[index]-np.sum(x[index:index+int(time_windows)])/time_windows)
tmp = np.array(tmp)

l = list()
for index in range(len(tmp)):
    if (np.abs(tmp[index])<2):
        l.append(1)
    else:
        l.append(0)
        
l= np.array(l)
for index in range(len(l)-int(time_windows)):
    if np.sum(l[index:index+int(time_windows)])>time_windows/2:
        print("error at time "+str(index))
        

plt.figure()
plt.plot(l)
plt.grid()
plt.savefig("timewi")

plt.figure()
plt.plot(stn)
plt.grid()
plt.savefig("stn")
"""
