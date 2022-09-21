import numpy as np
from common_library import *

class particleFilter:
    """
    A particle filter implementation
    ...
    Attributes
    ----------
    x_init : np.array
        The initial state of the system
        
    f : function
        The update function of the system: u = f(x)
    
    P0 : function
        The initial P(Y|u), the initial probabability matrix of heaving Y as output with u as input.
        If no information about this lead it to be a uniform probability
        
    N : int
        The number of particle of the filter
        
    Methods
    -------
    
    updateParticleFilter : 
        gain the next state estimation with the current state of the filter
    
    
    """
    def __init__(self,x_init,f,P0,N=10,dim_x=4):
        """
        Inizialization of the filter with:
        
        x_init  : the initial input at time k=0
        f       : the update function
        P0      : the probability estimation function
        N       : the number of particles
        """
        # Initial weigth
        self.W = np.ones(N)*(1/N)
        
        # TODO: vectorize this
        x = []
        for i in range(N):
            x.append(x_init)
            
        self.x = np.array(x)
        self.f = f
        self.P = P0
        self.N = N
        self.dim_x=dim_x
        
        self.W_list = []
        self.u_list = []
        self.noise_intensity = 0.0001
        
        return
    
    
    def noise(self,u,debug=False):
        """
        Appling the noise to the state as line 9 of the algoritm with the noise = 1 generated with probability = p
        """
        
        """
        p = self.noise_intensity
        #print(u[0:2**self.dim_x])
        state = getSubstates(u[0:2**self.dim_x])
        #u = getSubstates(u)

        n = np.random.choice([1,0], len(state), p=[p,1-p])
        
        if debug:
            print("xor")
            print(state)
            print(n)
        x = state.astype(int)^n.astype(int) # component-wise xor
        # Conatenate update part -> may need to put random I do not know
        x = np.concatenate((getState(x),u[2**self.dim_x:]))
        
        return x
        """
        return u
    
    def get_result(result):
        global results
        results.append(result)

    def computeUV(x,w,ui,y,f,p):
        u = f(x, ui)[0]
        p = p(y,u,ui)
        V = p * w[index]
        
        return (u,v)
        

    def updateParticleFilter(self,y,ui, debug=False):
        """
        Update the particle filter with the following input:
        
        ui : the input state at the time k-1
        y  : the output state at the time k
        """
        
        u = []
        V = []

        tmp = 0
        epoch = 0
        self.noise_intensity=0.04
        
        while(tmp == 0):
            u = []
            V = []

            epoch +=1
            """
            if (epoch == 2):
                print("Y:")
                print(y)
                # This is a sort of "hack" since if the first time that the computation
                # lead to a uncunclusive response to the probability, it retry to give back
                # some probability to some particle that have reached the zero weigth
                self.W += np.ones(self.N)*(1/self.N)*10**-(8)
                print("Error reached")
                if (self.noise_intensity < 0.5):
                    self.noise_intensity *= 2
                else:
                    self.noise_intensity = 0.5
            """
            if (epoch > 20):
                print("System crashed")

            assert epoch < 10, "Number of epoch exploded"
            
            for index in range(self.N):
                """
                Recall self.x is the estimate state at the time k-1
                ui is the input at the time k-1
                y is the outpu at the time k
                
                TODO: put this in multithread
                """
                next_state = self.f(self.x[index], ui)[0]
                # print(next_state)
                u.append( next_state )
                if debug:
                    print("u")
                    print(u[index])
                
                p = self.P(y,u[index],ui)
                if debug:
                    print("p")
                    print(p)
                V.append( p * self.W[index] + 1e-8)
                
            self.u_list.append(u)
            
            u = np.array(u)
            V = np.array(V) 
            if debug:
                print("u")
                print(u)
                print("v")
                print(V)
            tmp = np.sum(V)
            # resample, aka the new index of the u-vector
            #if(sum(V)==0):
            #    print(self.x)
            #    print(ui)
            #    print(y)
            #    print(u)
            #    print(V)
            #    print(self.P(y,u[index],ui))
        
        if (np.sum(V)==0):
            print( "V=0" )
        V = V/np.sum(V)
        if debug:
            print(V)
        
        
        mu = np.random.choice( np.linspace(0,self.N-1,self.N), self.N, p=V )
        if debug:
            print("mu")
            print(mu)
        
        # create the new value of x 
        x = []
        tilde_W = []
        t = 0
        for index in range(self.N):
            # TODO: put this in multithread
            ii = int(mu[index])
            x.append(self.noise(u[ii]))
            #print(x[index])
            P = self.P(y,u[ii],ui)
            if ( P != 0 ):
                try:
                    tilde_W.append( self.P(y,x[index],ui)/ P + 1e-8)
                except:
                    tilde_W.append( 1e-8)
                    print(x[index])
            else:
                #if debug:
                #print("particle impossible to have")
                t = t + 1
                tilde_W.append(self.P(y,x[index],ui) + 1e-8)
                #tilde_W.append(0)
        if (t>0):
            print(t)
        
        # save the current time state estimation
        self.x = np.array(x)
        
        # Recalculate the weigth and save them
        tilde_W = np.array(tilde_W)
        sum_tilde_W = np.sum(tilde_W)
        self.W = tilde_W/sum_tilde_W
        
        self.W_list.append(self.W)

        # calculate the estimate state
        z = 0
        # TODO: vectorize this 
       
        if debug:
            print("w")
            print(self.W)
        for i in range(self.N):
            z += self.W[i] * self.x[i][0:2**self.dim_x]
        
        if debug:
            print(z)
        # not sure if this is enogth or better to do 1-z or stuff like that
        return z
