import numpy as np
import tensorflow as tf
from custom_lstm import CustomBasicLSTMCell
"""
Actor Parameters
"""
TAU = 0.001
LEARNING_RATE= 0.001

GAMMA = 0.99
HIDDEN_UNITS = 250 #no. of hidden units in lstm cell
N_LAYERS = 2
TARGET_VALUE_DIMENSION = 1

class ActorNet:
    """ Actor Neural Network model of the RDPG algorithm """
    def __init__(self,N_STATES, N_ACTIONS,MAX_STEP,BATCH_SIZE):
        self.N_STATES = N_STATES        
        self.N_ACTIONS = N_ACTIONS        
        self.MAX_STEP = MAX_STEP
        self.BATCH_SIZE = BATCH_SIZE
        self.g=tf.Graph()
        with self.g.as_default():
            self.sess = tf.InteractiveSession()
            
            """
            Actor network:
            """
            self.a_input_states = tf.placeholder("float", [None, self.MAX_STEP, self.N_STATES],name='input_placeholder') 
            self.a_grad_from_critic = tf.placeholder("float", [None, self.MAX_STEP, self.N_ACTIONS],name='input_placeholder')

            self.W_a = tf.Variable(tf.random_normal([HIDDEN_UNITS, self.N_ACTIONS]))
            self.B_a = tf.Variable(tf.random_normal([1],-0.003,0.003))
            #lstms
            with tf.variable_scope('actor'):            
                self.lstm_cell = CustomBasicLSTMCell(HIDDEN_UNITS) #basiclstmcell modified to get access to cell weights
                self.lstm_layers = [self.lstm_cell]*N_LAYERS
                self.lstm_cell = tf.nn.rnn_cell.MultiRNNCell(self.lstm_layers,state_is_tuple = True)
                self.init_state = self.lstm_cell.zero_state(self.BATCH_SIZE,tf.float32)
                self.lstm_outputs, self.final_state = tf.nn.dynamic_rnn(self.lstm_cell, self.a_input_states,initial_state = self.init_state, dtype=tf.float32,sequence_length=self.length(self.a_input_states))
            self.lstm_outputs_list = tf.transpose(self.lstm_outputs, [1, 0, 2])
            self.lstm_outputs_list = tf.reshape(self.lstm_outputs_list, [-1, HIDDEN_UNITS])
            self.lstm_outputs_list = tf.split(0, self.MAX_STEP, self.lstm_outputs_list)
            #prediction(output) at each time step:(list of tensors)
            self.pred_t = [ tf.matmul(self.lstm_output, self.W_a) + self.B_a for self.lstm_output in self.lstm_outputs_list]
            self.pred_t_array = tf.pack(self.pred_t)
            self.pred_t_array = tf.transpose(self.pred_t_array, [1, 0, 2]) #(to get shape of batch sizexstepxdimension)
            """
            last relevant action (while evaluating actor during testing)
            """
            self.last_lstm_output = self.last_relevant(self.lstm_outputs,self.length(self.a_input_states))
            self.action_last_state = tf.matmul(self.last_lstm_output, self.W_a) + self.B_a 
            #optimizer:
            #self.params = tf.trainable_variables()
            self.params = [self.lstm_layers[0].weights,self.lstm_layers[0].bias,self.lstm_layers[1].weights,self.lstm_layers[1].bias,self.W_a,self.B_a]            
            self.a_grad_from_criticT = tf.transpose(self.a_grad_from_critic,perm=[1,0,2])
            self.gradient = tf.gradients(tf.pack(self.pred_t),self.params,-self.a_grad_from_criticT/(self.MAX_STEP * BATCH_SIZE))#- because we are interested in maximization
            self.opt = tf.train.AdamOptimizer(LEARNING_RATE)
            self.optimizer = self.opt.apply_gradients(zip(self.gradient,self.params))
            print("Initialized Actor Network...")
            
            """
            Target Actor network:
            """        
            self.t_a_input_states = tf.placeholder("float", [None, self.MAX_STEP, self.N_STATES],name='input_placeholder') 
            self.t_a_grad_from_critic = tf.placeholder("float", [None, self.MAX_STEP, self.N_ACTIONS],name='input_placeholder')

            self.t_W_a = tf.Variable(tf.random_normal([HIDDEN_UNITS, self.N_ACTIONS]))
            self.t_B_a = tf.Variable(tf.random_normal([1],-0.003,0.003))
            #lstms
            with tf.variable_scope('target_actor'):
                self.t_lstm_cell = CustomBasicLSTMCell(HIDDEN_UNITS) #basiclstmcell modified to get access to cell weights
                self.t_lstm_layers = [self.t_lstm_cell]*N_LAYERS
                self.t_lstm_cell = tf.nn.rnn_cell.MultiRNNCell(self.t_lstm_layers,state_is_tuple = True)
                self.t_init_state = self.lstm_cell.zero_state(self.BATCH_SIZE,tf.float32)
                self.t_lstm_outputs, self.t_final_state = tf.nn.dynamic_rnn(self.t_lstm_cell, self.t_a_input_states,initial_state = self.t_init_state, dtype=tf.float32,sequence_length=self.length(self.t_a_input_states))
            self.t_lstm_outputs_list = tf.transpose(self.t_lstm_outputs, [1, 0, 2])
            self.t_lstm_outputs_list = tf.reshape(self.t_lstm_outputs_list, [-1, HIDDEN_UNITS])
            self.t_lstm_outputs_list = tf.split(0, self.MAX_STEP, self.t_lstm_outputs_list)
            #prediction(output) at each time step:(list of tensors)
            self.t_pred_t = [tf.matmul(self.t_lstm_output, self.t_W_a) + self.t_B_a for self.t_lstm_output in self.t_lstm_outputs_list]
            self.t_pred_t = tf.pack(self.t_pred_t)
            self.t_pred_t = tf.transpose(self.t_pred_t, [1, 0, 2]) #(to get shape of batch sizexstepxdimension)
            """
            last relevant action (while evaluating actor during testing)
            """
            self.t_last_lstm_output = self.last_relevant(self.t_lstm_outputs,self.length(self.t_a_input_states))
            self.t_action_last_state = tf.matmul(self.t_last_lstm_output, self.t_W_a) + self.t_B_a
            print("Initialized Target Actor Network...")
            self.sess.run(tf.initialize_all_variables())
            
            #To initialize critic and target with the same values:
            # copy target parameters
            self.sess.run([
				self.t_lstm_layers[0].weights.assign(self.lstm_layers[0].weights),
				self.t_lstm_layers[0].bias.assign(self.lstm_layers[0].bias),
				self.t_lstm_layers[1].weights.assign(self.lstm_layers[1].weights),
				self.t_lstm_layers[1].bias.assign(self.lstm_layers[1].bias),
				self.t_W_a.assign(self.W_a),
				self.t_B_a.assign(self.B_a)
			])


            self.update_target_actor_op = [
                self.t_lstm_layers[0].weights.assign(TAU*self.lstm_layers[0].weights+(1-TAU)*self.t_lstm_layers[0].weights),
                self.t_lstm_layers[0].bias.assign(TAU*self.lstm_layers[0].bias+(1-TAU)*self.t_lstm_layers[0].bias),
                self.t_lstm_layers[1].weights.assign(TAU*self.lstm_layers[1].weights+(1-TAU)*self.t_lstm_layers[1].weights),
                self.t_lstm_layers[1].bias.assign(TAU*self.lstm_layers[1].bias+(1-TAU)*self.t_lstm_layers[1].bias),
                self.t_W_a.assign(TAU*self.W_a+(1-TAU)*self.t_W_a),
                self.t_B_a.assign(TAU*self.B_a+(1-TAU)*self.t_B_a)
            ]
                   
    def length(self,data):
        used = tf.sign(tf.reduce_max(tf.abs(data), reduction_indices=2))
        length = tf.reduce_sum(used, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        return length           

    def last_relevant(self,output, length): #method used while evaluating target net: where input is one or few time steps
        L_BATCH_SIZE = tf.shape(output)[0]
        max_length = int(output.get_shape()[1])
        out_size = int(output.get_shape()[2])
        index = tf.range(0, L_BATCH_SIZE) * max_length + (length - 1)
        flat = tf.reshape(output, [-1, out_size])
        relevant = tf.gather(flat, index)
        return relevant 
    
    def train_actor(self,o_n_t,del_Q_a):
        self.sess.run(self.optimizer, feed_dict={ self.a_input_states: o_n_t, self.a_grad_from_critic: del_Q_a})    

    def evaluate_actor(self,o_n_t):
        return self.sess.run(self.action_last_state, feed_dict={self.a_input_states: o_n_t})[0]    
    
    def evaluate_actor_batch(self,o_n_t):
        return self.sess.run(self.pred_t_array, feed_dict={self.a_input_states: o_n_t})   
        
    def evaluate_target_actor(self,o_n_t):
        return self.sess.run(self.t_pred_t, feed_dict={self.t_a_input_states: o_n_t})   
        
    def update_target_actor(self):
        self.sess.run(self.update_target_actor_op)        
        