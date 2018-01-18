'''
RDPG on continous system
'''
import numpy as np
import gym
from gym.spaces import Box, Discrete
from rdpg import RDPG
from ou_noise import OUNoise
import random
from time import sleep

U_MIN = -3
U_MAX = +3
#Parameters
NUM_ACTIONS = 1
NUM_OUTPUTS = 4 

EPISODES=10000

BATCH_SIZE = 32
experiment= 'InvertedPendulum-v1'
env= gym.make(experiment)

STEPS = 200#env.spec.timestep_limit #steps per episode 
N_STATES = env.observation_space.shape[0] #Number of observed states
N_ACTIONS = env.action_space.shape[0] #Number of input states
assert isinstance(env.observation_space, Box), "observation space must be continuous"
assert isinstance(env.action_space, Box), "action space must be continuous"
exploration_noise = OUNoise(1) #1 is number of input
agent = RDPG(env, N_STATES, N_ACTIONS,STEPS,BATCH_SIZE) #initialize critic, actor and corresponding target nets
exploration_noise = OUNoise(N_ACTIONS) 
reward_st = np.array([0])
R = []
for i in xrange(EPISODES):
    print "==== Starting episode no:",i,"====","\n"
    o_t = env.reset()
    reward_per_episode = 0
    for t in xrange(STEPS):
        
        a_t = agent.evaluate_actor(o_t,t) + exploration_noise.noise()
        #a_t = random.uniform(U_MIN,U_MAX)        
        print "Action at step", t ," :",a_t,"\n"        
        o_t1, r_t, done, info = env.step(a_t)
        #r_t = t #remove this
        #print r_t
        o_t = np.reshape(o_t,[1,N_STATES])
        a_t = np.reshape(a_t,[1,N_ACTIONS])
        r_t = np.reshape(r_t,[1,1])
        if t == 0:
            #initializing history at time, t = 0
            
            h_t = np.hstack([o_t,a_t,r_t])
        else:
            h_t = np.append(h_t,np.hstack([o_t,a_t,r_t]),axis = 0)
        reward_per_episode += r_t
        #appending history:
                    
        o_t = o_t1
        if (done or (t == STEPS-1)):
                print 'EPISODE: ',i,' Steps: ',t,' Total Reward: ',reward_per_episode
                print "Printing reward to file"
                exploration_noise.reset() #reinitializing random noise for action exploration
                reward_st = np.append(reward_st,reward_per_episode)
                np.savetxt('episode_reward.txt',reward_st, newline="\n")
                print '\n\n'
                agent.add_to_replay(h_t,i)
                break
                
    if i == 0:
        #store episodes:
        R.append(h_t)
        #R = np.zeros([1,STEPS,NUM_ACTIONS+NUM_OUTPUTS+1])
        #R = np.append(R,np.reshape(h_t,[1,STEPS,NUM_ACTIONS+NUM_OUTPUTS+1]),axis = 0)
        #R = np.delete(R, (0), axis=0) #Initialing a zero array with size and deleting it back

    else:
        R.append(h_t)
        #store sequences (ot,at,rt) in R
        #R = np.append(R,np.reshape(h_t,[1,STEPS,NUM_ACTIONS+NUM_OUTPUTS+1]),axis = 0)
    
    #train critic and actor network
    if(i>BATCH_SIZE+1):
#        agent.train()
        try:
            agent.train()
        except:
            print "Error Caught \n"
            continue


state_matrix = np.zeros([BATCH_SIZE,STEPS,N_STATES])
