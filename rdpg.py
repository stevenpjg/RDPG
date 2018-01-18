import numpy as np
from actor_net import ActorNet
from critic_net import CriticNet

BUFFER_SIZE = 1000
GAMMA = 0.99
class RDPG:
    """Recurrent Policy Gradient Algorithm"""
    
    def __init__(self,env, N_STATES, N_ACTIONS,STEPS,BATCH_SIZE):
        self.N_STATES = N_STATES
        self.N_ACTIONS = N_ACTIONS        
        self.STEPS = STEPS
        self.BATCH_SIZE = BATCH_SIZE #mini batch size        
        self.critic_net = CriticNet(self.N_STATES,self.N_ACTIONS,self.STEPS,self.BATCH_SIZE)
        self.actor_net = ActorNet(self.N_STATES,self.N_ACTIONS,self.STEPS,self.BATCH_SIZE)
        self.R = []
    def evaluate_actor(self,state,t):
        #converting state to a 3D tensor to feed into lstms
        if t==0:
            self.state_matrix = np.zeros([self.BATCH_SIZE,self.STEPS,self.N_STATES])
            self.state_matrix[0,t,:] = state        
        else:
            self.state_matrix[0,t,:] = state
#        print self.state_matrix
#        raw_input('Enter to continue')
        return self.actor_net.evaluate_actor(self.state_matrix)
    
    def add_to_replay(self,h_t,i):
        ##STORE THE SEQUENCE (o_1,a_1,r_1,....,o_t,a_t,r_t) in R
        self.h_t = h_t        
        self.R.append(h_t)
        if(len(self.R)>BUFFER_SIZE):
            self.R.pop(0)

        
    def sample_mini_batches(self):
        self.indices = np.random.randint(1,len(self.R),size=(1,self.BATCH_SIZE))
        self.R_mini_batch = [None] * self.BATCH_SIZE
        for i in range(0,len(self.indices[0,:])):
            self.R_mini_batch[i]=self.R[self.indices[0][i]] 
        
        
        
        #reward_t (batchsize x timestep)
        self.r_n_tl = [None] * self.BATCH_SIZE
        for i in range(0,len(self.r_n_tl)):
            self.r_n_tl[i] = self.R_mini_batch[i][:,-1]

        self.r_n_t = np.zeros([self.BATCH_SIZE,self.STEPS])

        for i in range(0,self.BATCH_SIZE):
            self.r_n_t[i,0:len(self.r_n_tl[i])] = self.r_n_tl[i]
    
        #observation list (batchsize x timestep)
        self.o_n_tl = [None] * self.BATCH_SIZE
        for i in range(0,len(self.o_n_tl)):
            self.o_n_tl[i] = self.R_mini_batch[i][:,0:self.N_STATES]   

        self.o_n_t = np.zeros([self.BATCH_SIZE,self.STEPS,self.N_STATES])    
        for i in range(0,self.BATCH_SIZE):
            self.o_n_t[i,0:len(self.o_n_tl[i]),:] = self.o_n_tl[i]


        #action list:    
        #observation list (batchsize x timestep)
        self.a_n_tl = [None] * self.BATCH_SIZE
        for i in range(0,len(self.a_n_tl)):
            self.a_n_tl[i] = self.R_mini_batch[i][:,self.N_STATES:self.N_STATES+self.N_ACTIONS]       

        self.a_n_t = np.zeros([self.BATCH_SIZE,self.STEPS,self.N_ACTIONS])    
        for i in range(0,self.BATCH_SIZE):
            self.a_n_t[i,0:len(self.a_n_tl[i]),:] = self.a_n_tl[i]

    def train(self):
        self.sample_mini_batches()
        #Action at h_t+1:
        self.t_a_ht1 = self.actor_net.evaluate_target_actor(self.o_n_t)
        #State Action value at h_t+1:
        
        self.t_qht1 = self.critic_net.evaluate_target_critic(self.o_n_t,self.t_a_ht1)
        self.check = self.t_qht1
       
        ##COMPUTE TARGET VALUES FOR EACH SAMPLE EPISODE (y_1,y_2,....y_t) USING THE RECURRENT TARGET NETWORKS
        self.y_n_t = []
        self.r_n_t = np.reshape(self.r_n_t,[self.BATCH_SIZE,self.STEPS,1])

        for i in range(0,self.STEPS):
            if (i == self.STEPS-1):    
                self.y_n_t.append(self.r_n_t[:,i])
            else:    
                self.y_n_t.append(self.r_n_t[:,i,:] + GAMMA * self.t_qht1[:,i+1,:])
        self.y_n_t = np.vstack(self.y_n_t)
        self.y_n_t = self.y_n_t.T #(batchsize x timestep)
        self.y_n_t = np.reshape(self.y_n_t,[self.BATCH_SIZE,self.STEPS,1]) #reshape y_n_t to have shape (batchsize,timestep,no.dimensions) 
        ##COMPUTE CRITIC UPDATE (USING BPTT)
        self.critic_net.train_critic(self.o_n_t,self.a_n_t,self.y_n_t)
        
        #action for computing critic gradient
        self.a_ht = self.actor_net.evaluate_actor_batch(self.o_n_t) #returns output as 3d array
        #critic gradient with respect to action delQ/dela
        self.del_Q_a = self.critic_net.compute_critic_gradient(self.o_n_t,self.a_ht)
        ##COMPUTE ACTOR UPDATE (USING BPTT)
        self.actor_net.train_actor(self.o_n_t,self.del_Q_a)
        ##Update the target networks
        self.critic_net.update_target_critic()
        self.actor_net.update_target_actor()
        
        