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


def print_data(x,alpha):
    plt.figure(figsize=(15,6))
    print_evolution_state(axe, PBN['x,u,y'][0])
    plt.savefig("PBNc_states.png")
            
    plt.figure(figsize=(15,6))
    plt.plot(alpha)
    plt.xlabel("time")
    plt.ylabel("alpha")
    plt.savefig("alpha.png")
            


steps = 300
particles = 1000

# The network to simulate
namefile="edkcbr_sano"
# The PBN network to use
exec(open("../networks/"+namefile+".json").read())



print("Load data")
# Normal data
ax  = np.load("ax.npy", allow_pickle=True)
ay  = np.load("ax.npy", allow_pickle=True)
au  = np.load("au.npy", allow_pickle=True)


axe = []
axe.append(ax[0])

pc1 = cascadeLPBN(PBN, au,namefile)
pc_cancer = cascadeLPBN(PBNc, u,error_file)
pf = particleFilter(ax[0],pc1.updateSystem,pc1.p,N=particles,dim_x=PBN['x,u,y'][0])
pf_c = particleFilter(x0,pc_cancer.updateSystem,pc_cancer.p,N=particles,dim_x=PBNc['x,u,y'][0])


alpha = []
alpha.append(1)

x = ax[0]

# Do the simulation with the estimated L
tin = time.time()
for i in range(1,steps):
        try:
    
            print("step: "+str(i))
            
            # Simulate the particle filter
            if(i == 1):
                pc1.set_previus_u=au[i-1]
            else:
                pc1.set_previus_u=au[i-2]
            # particle filter at time k-1, need the k at time k and y at time k-1
            
            
            xe = pf.updateParticleFilter(ax[i], au[i-1])
            xe_c = pf_c.updateParticleFilter(ax[i], au[i-1])


            print(xe)
    
            #xe = pc1.estimateSystem(x,au[i-1])
            
            x_tmp = (xe*ax[i][0:2**(PBN['x,u,y'][0])] )
            current_alpha = np.sum(x_tmp)
            
            x_tmp_c = (xe_c*ax[i][0:2**(PBN['x,u,y'][0])] )
            current_alpha_c = np.sum(x_tmp_c)
                        
            print("current alpha "+str(current_alpha))
            print("current alpha "+str(current_alpha_c))

            shel = max(current_alpha,current_alpha_c)
            
            if (shel == 0):
                x = np.zeros(len(x_tmp)) * 1e-8
            elif (shel == current_alpha):
                x = x_tmp/shel
            else:
                x = x_tmp_c/shel
            
            axe.append(x)
            
            alpha.append(current_alpha)
            alpha_c.append(current_alpha_c)

            
            print_data(axe,alpha)
            
        except:
            print("Errore")
            traceback.print_exc()
        
tfin = time.time()
print("Estimation by kf total time ="+str(tfin-tin))

alpha = np.array(alpha) 

print_data(axe,alpha)

