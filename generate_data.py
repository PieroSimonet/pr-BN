#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
from math import floor, log
import os
from datetime import date
import signal

# my library
from common_library import *
from particle_filter import *
from cascadeLPBN import *
from print_library import *

import time

def generate_folder(path):
  if not os.path.exists(path):
      os.makedirs(path)

# Close the generation data      
def handler(signum, frame):
    res = input("Ctrl-c was pressed. Do you really want to exit? y/n ")
    if res == 'y':
        exit(1)
 
signal.signal(signal.SIGINT, handler)
  
# The simulation "time"
steps = 300
# When appling the damage
damage_time = 200
end_damage_time = steps

# The network to simulate
namefile="edkcbr_sano"
# The network with the cancer-typed error
error_file="edkcbr_cancer"

# The PBN network to use
exec(open("../networks/"+namefile+".json").read())
# The PBNc (PBN of the cancer) network to use
exec(open("../networks/"+error_file+".json").read())

generation = 160

while True:
  print("Generation "+str(generation))
  # Generte an initial state 
  x01 = getRandState(PBN['x,u,y'][0])
  x02 = getRandState(PBN['x,u,y'][0])
  x0 = np.concatenate((x01,getXD(x01,x02)))
  u = getRandState(PBN['x,u,y'][1])
  # x0 = np.array([0,1,0,0,0,1])
  # u = np.array([0,1])
  # print("Models creation...")
  pc1 = cascadeLPBN(PBN, u,namefile)
  pc_cancer = cascadeLPBN(PBNc, u,error_file)
  # pc1.gel() # generate the estimated matrix for the normal behaviur
  # print("Done")
  # y time 0, u time 0, x time 1
  # print("System inizialization...")
  xc,yc  = pc1.updateSystem(x0,u)
  # print("Done")
  
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
          # Broken network
          xc,yc  = pc_cancer.updateSystem(xc,u,True)
      else:
          # Normal network
          xc,yc  = pc1.updateSystem(xc,u,True)
      ay.append(yc)
      if (i % 10 == 0):
        print("time :"+str(i))   
         
  tfin = time.time()
  print("Simulation time ="+str(tfin-tin))
  
  data_str = "data/"+str(date.today())+"_"+str(generation)
  generate_folder(data_str)
  np.save(data_str+"/ax", ax)
  np.save(data_str+"/au", au)
  np.save(data_str+"/ay", ay)
  
  print(ax)
  
  plt.figure(figsize=(15,6))
  print_evolution_state(ax, PBN['x,u,y'][0])
  plt.savefig(data_str+"/states.png")
  
  generation = generation+1
 
  
 
  
