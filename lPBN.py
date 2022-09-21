import numpy as np
from math import floor,log
import multiprocess as mp
import pickle

# My custom library for common functions for boolean stuff
from common_library import *


num_process = mp.cpu_count() - 2

def get_result(result):
    global results
    results.append(result)
    
def mysort(vet):
    return vet[1]    

def calculateValue(u, dim_x, string, position, dx):
    vett = np.ones(dim_x)
    
    if (dim_x > 0):
        tmp = np.zeros(2**dim_x, dtype = int)
        tmp[dx]=1
        x= getSubstates(tmp)
                                    
    for index in range(len(string)):
        exec("vett[index]="+string[index])
    
    return np.argwhere(getState(vett)==1)[0][0], position

def next_step(matrix,index):
    L = np.array(np.matrix(matrix)[:,index].reshape(-1))[0]
    return np.argwhere(L==1)[0][0], index
    
def evaluate(string, dim_u,dim_x):
    
    u = np.zeros(dim_u, dtype = int)    
    x = np.zeros(dim_x, dtype = int)
    
    mat = list()

    for du in range(2**dim_u):
        if (dim_u > 0):
            tmp = np.zeros(2**dim_u, dtype = int)
            tmp[du]=1
            u = getSubstates(tmp)
            
        global results
        results = []
        
        pool = mp.Pool(num_process)
        
        for dx in range(2**dim_x):
            pool.apply_async(calculateValue, args=(u,dim_x, string, du+dx,dx),callback=get_result)
        
        pool.close()
        pool.join()
        results.sort(reverse=True, key=mysort)
        
        for dx in range(2**dim_x):
            mat.append(results.pop()[0])
    
    mat = np.array(mat)
    #print(mat)        
    return mat

class lPBN:     
    """
    A class to work with syncr probabilistic boolean networks
    
    ...
    Attributes
    ----------
    PBN : Dictionary
        a custom dictionary with the description of the PBN
        
    Methods
    -------
    
    -------
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
    
    """
    
    def __generateLH(self, L_strings, H_strings, dim_x, dim_u, dim_y):

        L = evaluate(L_strings, dim_u,dim_x)
        H = evaluate(H_strings, dim_u,dim_x)
            
        #L =  np.matrix(L).T  
        #H =  np.matrix(H).T  
        
        return L,H
        
    def gLH(self, L_strings, H_strings, dim_x, dim_u, dim_y):
        return self.__generateLH(L_strings, H_strings, dim_x, dim_u, dim_y)

    def __setCombinationStrings(self, element, label, position, slist,p):
        if (position == len(element)):
            self.element[label].append([np.array(slist), p])
            return
        
            
        for index in range(len(element[position])):
            l = slist.copy()
            l.append(element[position][index][0])
            self.__setCombinationStrings(element, label, position+1, l, p*element[position][index][1])
    
    def sc(self, element, label, position, slist,p):
        return self.__setCombinationStrings(element, label, position, slist,p)
     
            
    def __generateEstimateLH(self):
        
        #self.L = np.zeros(self.systems[0][0].shape)
        #self.H = np.zeros(self.systems[0][1].shape)
        self.L = 0
        self.H = 0        
        
        for index in range(len(self.probability)):
            L = []
            for i in range(len(self.systems[index][0])):
                L.append(self.get_vector_from_index(self.systems[index][0][i], self.PBN['x,u,y'][0] ))
            H = []
            L = np.matrix(L).T
            for i in range(len(self.systems[index][0])):
                H.append(self.get_vector_from_index(self.systems[index][1][i], self.PBN['x,u,y'][2] ))
            H = np.matrix(H).T

            self.L += L * self.probability[index][0]
            self.H += H * self.probability[index][0]
    
    def gel(self):
        try:
            file_L = open(self.namefile+"_L_systems.npy", "rb")
            file_H = open(self.namefile+"_H_systems.npy", "rb") 
            self.L = pickle.load(file_L)
            self.H = pickle.load(file_H)
            file_L.close()
            file_H.close()
        except:
            self.__generateEstimateLH()
            
            file_L = open(self.namefile+"_L_systems.npy", "wb")
            file_H = open(self.namefile+"_H_systems.npy", "wb") 
            pickle.dump(self.L, file_L)
            pickle.dump(self.H, file_H)
            file_L.close()
            file_H.close()
                  
 
    
    def estimateSystem(self,x,u):
        tmp = semi_tensor(self.L, u.reshape(-1,1))
        return  np.asarray(semi_tensor(tmp,x.reshape(-1,1)).reshape(-1) ) [0,:]
        
            
            
    def updateSystem(self, x, u):
        """
        x,u shuld be in state form, at the time k-1
        
        return an obejet of the form:
            [x,y]
            where:
             - x is at the time k
             - y is at the time k-1
        """
        ptmp=  self.probability[:,0]/np.sum( self.probability[:,0])
        index = int( np.random.choice(np.linspace(0,len(self.probability[:,0])-1,len(self.probability[:,0])), p = ptmp ) )
        
        
        if (u.size > 0):
            state = getState(np.concatenate((getSubstates(u),getSubstates(x))))
        else:
            state = x
            
        i = np.argwhere(state == 1)[0][0]
        
        H = self.systems[index][1]
        L = self.systems[index][0]
        y = self.get_vector_from_index(H[i],self.PBN["x,u,y"][2])
        x = self.get_vector_from_index(L[i],self.PBN["x,u,y"][0])
        #semi_tensor(semi_tensor(self.systems[index][1],u.reshape(1,-1)),x)
        
        #x = getState(np.concatenate((getSubstates(u),getSubstates(x))))
        
        #x = np.matmul(self.systems[index][0],x)
        
        return x,y#np.array(x)[0],np.array(y)[0]
    
    def getL(self):
        return self.L
    
    def getH(self):
        return self.H
    
    def p(self,y,x_in,u):
        """
        Give the state at the time k, the output of the system at the time k
        compute the probability P(Y|X), for now using E(Yh)=sum ph*H*xk 
        and evaluate the probability to be in that estate
        
        TODO: check the possibility to be in a "distance" of that output position
        """
        ey = self.pey(y,x_in,u)
        i = int(np.argwhere(y==1)[0][0])
        p = np.array(ey)[i]
        
        return p
        
    def pey(self,y,x_in,u):
        """
        Give the state at the time k, the output of the system at the time k
        compute the probability P(Y|X), for now using E(Yh)=sum ph*H*xk 
        and evaluate the probability to be in that estate
        
        TODO: check the possibility to be in a "distance" of that output position
        """
        x = x_in[0:2**self.PBN["x,u,y"][0]]
        ey = 0
        
        if (u.size > 0):
            state = getState(np.concatenate((getSubstates(u),getSubstates(x))))
        else:
            state = x
            
        i = np.argwhere(state == 1)[0][0]
        
        Hs = np.array(self.systems, dtype=object)[:,1,:]
                
        for index in range(len(Hs)):
            # print(self.systems[index][1])
            H = Hs[index]
            ey += self.get_vector_from_index(H[i],self.PBN["x,u,y"][2]) * self.probability[index,2]
        
        s = np.sum(ey)
        ey = ey/s
        
        # print(ey)

        return np.array(ey)
        
   
        
    def matrix_to_vector(self, matrix):
        
        global results
        results = []
        
        pool = mp.Pool(mp.cpu_count()-1)
        
        for index in range(matrix.shape[1]):
            pool.apply_async(next_step, args=(matrix,index),callback=get_result)

        pool.close()
        pool.join()
        results.sort(reverse=False, key=mysort)
                    
        L_vector = np.array(results)[:,0]

        return L_vector
    
    def get_vector_from_index(self,index, shape):
        index = int(index)
        v = np.zeros(2**shape)
        v[index] = 1
        return v
    
    def __init__(self, PBN, namefile):
        
        self.PBN = PBN
        
        self.namefile=namefile

        try:
            file_sys  = open(self.namefile+"_main_systems.npy", "rb")
            self.systems = pickle.load(file_sys)
            file_sys.close()
            file_prob = open(self.namefile+"_main_probability.npy", "rb") 
            self.probability = pickle.load(file_prob)
            file_prob.close()
            #self.__generateEstimateLH()
            self.L = 0
            self.H = 0

        except:
            self.element = {}
            self.element["L"] = list()
            self.element["H"] = list()
            
            l = list()
            
            self.systems = list()
            self.probability = list()
            
            self.__setCombinationStrings(PBN['x'],"L",0,l,1)
            self.__setCombinationStrings(PBN['y'],"H",0,l,1)
            
            for L in self.element["L"]:
                for H in self.element["H"]:
                    tL,tH = self.__generateLH(L[0], H[0], PBN['x,u,y'][0], PBN['x,u,y'][1], PBN['x,u,y'][2])
                    #vL = self.matrix_to_vector(tL)
                    #vH = self.matrix_to_vector(tH)
                    self.systems.append([tL,tH])
                    self.probability.append([L[1]*H[1],L[1],H[1]])
                    
            del(self.element)
            
            #self.__generateEstimateLH()
            self.L = 0
            self.H = 0

                
            
            self.probability = np.array(self.probability)
            
            file_sys  = open(self.namefile+"_main_systems.npy", "wb")
            pickle.dump(self.systems, file_sys) 
            file_sys.close()

            file_prob = open(self.namefile+"_main_probability.npy", "wb") 
            pickle.dump(self.probability, file_prob)
            file_prob.close()
            
        

