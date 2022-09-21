#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
from math import floor, log

from plyer import notification

# my library
from common_library import *
from particle_filter import *
from lPBN import *
from print_library import *


import time

# The simulation "time"
steps = 300
# When appling the damage
damage_time = 200
end_damage_time = steps
# This number should be sufficently big to preserve allow all
# the possible 
particles = 500

# The network to simulate
namefile="edkcbr_sano"
# The network with the cancer-typed error
error_file="edkcbr_cancer"

# The PBN network to use
exec(open("../networks/"+namefile+".json").read())
# The PBNc (PBN of the cancer) network to use
exec(open("../networks/"+error_file+".json").read())

# Generte an initial state 
x0 = getState(np.array([1,1,1,0,1,0,0,1,0,1,1,1])) # getRandState(PBN['x,u,y'][0])
u = getRandState(PBN['x,u,y'][1])
print("Models creation...")
pc1 = lPBN(PBN, namefile)
pc_cancer = lPBN(PBNc, error_file)
print("genLH...")
pc1.gel() # generate the estimated matrix for the normal behaviur
print("genLHc...")
pc_cancer.gel()

print("Done")
print("L:")
print(pc1.L.shape)
print("H:")
print(pc1.H)

print("--------------------------------------------")

print("L:")
print(pc_cancer.L)
print("H:")
print(pc_cancer.H)

# notification.notify("matrix created")


print("Particle Filer creation....")
pf = particleFilter(x0,pc1.updateSystem,pc1.p,N=particles,dim_x=PBN['x,u,y'][0])
pf_c = particleFilter(x0,pc_cancer.updateSystem,pc_cancer.p,N=particles,dim_x=PBNc['x,u,y'][0])

print("Done")




# y time 0, u time 0, x time 1
print("System inizialization...")
xc,yc  = pc1.updateSystem(x0,u)
print("Done")

# Inizialize the state vectors
ax = []
au = []
ay = []

ax.append(x0)
au.append(u)
ay.append(yc)

print("xc:"+str(x0))
print("yc:"+str(yc))

tin = time.time()

# Do the first simulation
print("---")
for i in range(steps):
    # Simulate the system
    # print("time :"+str(i))
    u = getRandState(PBN['x,u,y'][1])
    ax.append(xc)
    au.append(u)
    #ts = time.time()
    
    if (i > damage_time and i < end_damage_time):
        # start the stuck at damage
        #xc,yc  = pc1.updateSystem(xc,u,True)

        # xc = set_damage(xc,0,1,True)
        # Start the damage network
        xc,yc  = pc_cancer.updateSystem(xc,u)
    else:
        # Start the normal network
        # TODO: add a random state that have changed because of the break of the newtwork
        xc,yc  = pc1.updateSystem(xc,u)
    ay.append(yc)
    
tfin = time.time()
print("Simulation time ="+str(tfin-tin))
 

 
# Save the data simulated data
#np.save("ax", ax)
#np.save("au", au)
#np.save("ay", ay)

print(ax[0].shape)

fig = plt.figure(figsize=(15,6))
print_evolution_state(ax, PBN['x,u,y'][0])
plt.savefig("PBN_states.png")
  
 
# plt.figure()  
# state2graph(ay,PBN['x,u,y'][0] )
# plt.savefig("ay.png")


# y time 0, u time 0, x time 1

axe = []
axe.append(x0)


alpha = []
alpha.append(1)


alpha_c = []
alpha_c.append(0)


x = x0

# Do the simulation with the estimated L
tin = time.time()
for i in range(1,steps):
        try:
    
            print("step: "+str(i))
            
            xe = pf.updateParticleFilter(ax[i], au[i])  
            xe_c = pf_c.updateParticleFilter(ax[i], au[i])  

            
            #xe = pf.updateParticleFilter(ax[i], au[i])
            xe = pc1.estimateSystem(x,au[i-1])
            
            xe_c = pc_cancer.estimateSystem(x,au[i-1])


            print(xe)
    
            #xe = pc1.estimateSystem(x,au[i-1])
            
            x_tmp = xe*ax[i]
            current_alpha = np.sum(x_tmp)
            
            x_tmp_c = xe_c*ax[i]
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
            
        except:
            print("Errore")
        
tfin = time.time()
print("Estimation by L total time ="+str(tfin-tin))

plt.figure(figsize=(15,6))
print_evolution_state(axe, PBN['x,u,y'][0])
plt.savefig("PBNc_states.png")

alpha = np.array(alpha) 

plt.figure(figsize=(15,6))
plt.plot(alpha)
plt.xlabel("time")
plt.ylabel("alpha")
plt.savefig("alpha.png")


plt.figure(figsize=(15,6))
plt.plot(alpha_c)
plt.xlabel("time")
plt.ylabel("alpha")
plt.savefig("alpha_c.png")



"""

pf = particleFilter(x0,pc1.updateSystem,pc1.p,N=1000,dim_x=PBN['x,u,y'][0])


# Do the simulation with the particle filter
tin = time.time()
for i in range(1,steps):
    try:
        print("step: "+str(i))
        xe = pf.updateParticleFilter(ax[i], au[i])
        axe.append(xe)
    except:
        print("erore - not know what")
tfin = time.time()
print("Particele filter total time ="+str(tfin-tin))
#ae = []
#np.save("axe", axe)


plt.figure(figsize=(15,6))
print_evolution_state(axe, PBN['x,u,y'][0])
plt.savefig("PBNc_states.png")

"""

"""
    
# Calculate the error of the states
for i in range(0,steps):
    current_state = axe[i]
    x,p = get_all_substate(current_state, PBN['x,u,y'][0] )
    ey = np.zeros(2**PBN['x,u,y'][2])
    for index in range(len(x)):
        ey += pc1.pey(ay[i],x[index],au[i]) * p[index] 
    ae.append(ay[i]-ey)
    
# Save the error noise
np.save("ae", ae)

# Generate all the other figures
plt.figure()  
state2graph(ax,PBN['x,u,y'][0] )
state2graph(axe,PBN['x,u,y'][0] )
plt.savefig("a.png")
#plt.figure()
#state2graph(ay,4)
plt.figure()
state2graph(ae, PBN['x,u,y'][2]  )
plt.savefig("error.png")

exec(open("test_error.py").read())

"""
