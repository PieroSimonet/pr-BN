import numpy as np
from math import floor,log
from copy import deepcopy as dcopy
import pickle

from common_library import *
from lPBN import lPBN

class cascadeLPBN(lPBN):
    """
    Cascade upgrade for probabilistic boolean network, as the other class, but the mantency of the 
    XD values are inside the state V as follow: [x_1, .... x_n | xd ]^T 
    this is a necessary thing for simplify the particle filter class, since is possible to define
    V_{k+1} = F(V_k) carring both, the information about f and about u (look in my obisdin notes for reference)
    
    note: the length of the state is not static since xd as described in the common_library is not a boolean but 
    an array of the index of the state that have been updated in the previus iteration
    
    the input do not contain the difference between this and the previus, since is stored in a variable 
    insided the class that need to be initialized 
    
    -----
    The structure of the class is the following:
        - previus_u      : the previus input of the system
        - PBN            : the json description of the system
        - systems        : the vector of matrixs that desribe the system
            - [0] : L
            - [1] : H
        - probability    : the vector of the probability of the system
            - [0] : of the couple L/H at the correspective index of systems
            - [1] : L
            - [2] : H
        - Lxd            : a dictionary that contains in the position "uNUMBERSxNUMBERS" the corrispettive
                           systems[0] matrix updated to taking in account the ud and xd in action, described 
                           with the NUMBERS. Recall it sill have the probability of L[index]
    """

    def __calculateLXD(self, ud, xd):
        
        # xd, ud = vectors of the number of the node that have changed in the last iteration
        
        # Check that there was some change
        if (len(xd) == 0 and len(ud) == 0):
            return
        
        # current TODO: reviw all the following code with the new convenction
        # next TODO: save in a file the already checked configurations
        
        # Generate an array of the dimension of the number of state nodes and update and set it to one if 
        # the corresponding update function uses the node cited on xd,ud
        updated = np.zeros(self.PBN["x,u,y"][0]) 
        position_string = "u"+str(ud)+"x"+str(xd)
        
        if (not self.Lxd.get(position_string) == None):
            return
        
        
        # index = the node that have changed in the previus update
        for index in xd:
            # sub_index the index of the list of the possible nodes
            for sub_index in range(len(updated)):
                # only do if is not already checked that it will be updated
                if(not updated[sub_index]):
                    # now check for each node all the possible update functions
                    for sub_sub_index in range(len(self.PBN["x"][sub_index])):
                        if ( self.PBN["x"][sub_index][sub_sub_index][0].find( "x["+str(index)+"]" ) >= 0 ):
                            updated[index] = 1
                            
        # Basicaly it redo the same of xd with ud
        for index in ud:
            for sub_index in range(len(updated)):
                if(not updated[sub_index]):
                    for sub_sub_index in range(len(self.PBN["x"][sub_index])):
                        if ( self.PBN["x"][sub_index][sub_sub_index][0].find( "u["+str(index)+"]" ) >= 0):
                            updated[index] = 1
                               
        # Get a copy of the current PBN configuration                    
        xd_PBN = dcopy(self.PBN)
        
        # For the nodes that have no dependency in the previus update change all the possible update function to the identity
        for index in range(len(updated)):
            if(not updated[index]):
                xd_PBN["x"][index] = [["x["+str(index)+"]",1]]
                
        # Generate the new matrixs for the update
        self.element = {}
        self.element["L"] = list()
        self.element["H"] = list()
        
        l = list()
        
        self.Lxd["u"+str(ud)+"x"+str(xd)] = list()
        self.probability_L["u"+str(ud)+"x"+str(xd)] = list()
        
        self.sc(self.PBN['x'],"L",0,l,1)
        self.sc(self.PBN['y'],"H",0,l,1)
        
        for L in self.element["L"]:
            for H in self.element["H"]:
                tL,tH = self.gLH(L[0], H[0], self.PBN['x,u,y'][0], self.PBN['x,u,y'][1], self.PBN['x,u,y'][2])
                #vL = self.matrix_to_vector(tL)
                #vH = self.matrix_to_vector(tH)
                self.Lxd["u"+str(ud)+"x"+str(xd)].append([tL,tH])
                self.probability_L["u"+str(ud)+"x"+str(xd)].append([L[1]*H[1],L[1],H[1]])
                
        del(self.element)
        self.probability_L["u"+str(ud)+"x"+str(xd)] = np.array(self.probability_L["u"+str(ud)+"x"+str(xd)])
        
        
        file_sys  = open(self.namefile+"_systems.npy", "wb")
        file_prob = open(self.namefile+"_probability.npy", "wb")         
        pickle.dump(self.Lxd, file_sys)
        pickle.dump(self.probability_L, file_prob)
        file_sys.close()
        file_prob.close()
    
        return
    
    def updateSystem(self, x, u,update_u = False):
    
        # generate the difference of the input and the previus one
        ud = getXD(u, self.previus_u)
        if (update_u):
            self.set_previus_u(u)
        # extract the diffence  update on the state
        in_xd = 2 ** self.PBN["x,u,y"][0]
        xd = x[in_xd:len(x)]
        current_state = x[0:in_xd]
        
        if (len(xd)>0 or len(ud)>0):
            # generate the update matrixs for the current cascade state
            self.__calculateLXD(ud,xd)
            #print("u"+str(ud)+"x"+str(xd))
            # Get the random index for the update
            index = int( np.random.choice(np.linspace(0,len(self.probability_L["u"+str(ud)+"x"+str(xd)][:,0])-1,len(self.probability_L["u"+str(ud)+"x"+str(xd)][:,0])), p = self.probability_L["u"+str(ud)+"x"+str(xd)][:,0]) )
            # Generate the output of the previus timewindow
            full_state = getState(np.concatenate((getSubstates(u),getSubstates(current_state))))
                 
            i = np.argwhere(full_state == 1)[0][0]
            
            H = self.Lxd["u"+str(ud)+"x"+str(xd)][index][1]
            L = self.Lxd["u"+str(ud)+"x"+str(xd)][index][0]
            
            y = self.get_vector_from_index(H[i],self.PBN["x,u,y"][2])
            next_x = self.get_vector_from_index(L[i],self.PBN["x,u,y"][0])
            
            # x = semi_tensor(semi_tensor(L,u),self.last_x)
            xd = getXD(next_x, current_state)
            
            x = np.concatenate((next_x,xd))
        else:
            # Generate only the output
            y = super().updateSystem(x,u)[1]

        y = np.array(y).reshape(-1)

        return x,y 
                
    def set_previus_u(self, u):
        self.previus_u = u
        
    def get_previus_u(self):
        return self.previus_u
    
    def __init__(self, PBN, initial_u,namefile):
        # Inizialize the normal PBN
        super().__init__(PBN,namefile)
        # Create the dictionary with the possible update funtion L_{UD,XD} and the corrispettive probability
        try:
            file_sys  = open(self.namefile+"_systems.npy", "rb")
            file_prob = open(self.namefile+"_probability.npy", "rb") 
            self.Lxd = pickle.load(file_sys)
            self.probability_L = pickle.load(file_prob)
            file_sys.close()
            file_prob.close()
        except:
            self.probability_L = {}
            self.Lxd = {}
        # Set the initial input of the system
        self.set_previus_u(initial_u)        

