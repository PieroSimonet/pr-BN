#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
from math import floor, log

# my library
from common_library import *
from particle_filter import *
from cascadeLPBN import *
from print_library import *

import time

# Data folder
folder = "data/2022-05-27_12/"

# The network to simulate
namefile="edkcbr_sano"
# The network with the cancer-typed error
# error_file="edkcbr_cancer"

# The number of particle for the particle filter
particles = 350

ax = np.load(folder+"ax.npy", allow_pickle=True)
au = np.load(folder+"au.npy", allow_pickle=True)
ay = np.load(folder+"ay.npy", allow_pickle=True)

plt.figure(figsize=(15,6))
print_evolution_state(ax, 12)
plt.savefig("jamaica.png")

# Create the network
exec(open("../networks/"+namefile+".json").read())

print("start creation filter")

pc1 = cascadeLPBN(PBN, au[0],namefile)
pf = particleFilter(ax[0], pc1.updateSystem, pc1.p, N=particles, dim_x=PBN['x,u,y'][0])

# Execute the filter
axe = []
axe.append(ax[0])
  
print("end creation filter")
    
# Do the simulation with the particle filter
tin = time.time()
for i in range(1,len(ax)):
    try:
        print("step: "+str(i))
        # Simulate the particle filter
        if(i == 1):
            pc1.set_previus_u=au[i-1]
        else:
            pc1.set_previus_u=au[i-2]
        # particle filter at time k-1, need the k at time k and y at time k-1
        xe = pf.updateParticleFilter(ay[i], au[i-1])
        axe.append(xe)
    except:
        print("erore - not know what")
tfin = time.time()

print("Particele filter total time ="+str(tfin-tin))

np.save(folder+"axe", axe)

plt.figure(figsize=(15,6))
print_evolution_state_bis(axe, PBN['x,u,y'][0])
plt.savefig(folder+"/filter_states.png")
