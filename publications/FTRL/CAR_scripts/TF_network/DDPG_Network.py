
import tensorflow as tf
import numpy as np

def lrelu(x, leak=0.2, name='lrelu'):
	return tf.maximum(x, leak*x)

class DDPG(object):
    TAU = 0.01
    GAMMA = 0.99
    LR_A = 0.00001
    LR_C = 0.00001
    def __init__(self, sess, a_dim, s_dim, abound):
        self.pointer = 0
        self.sess = sess

        self.a_dim, self.s_dim, self.abound = a_dim, s_dim, abound
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')
        self.action_lower_bound = tf.constant(abound[0].reshape(1,-1),dtype=tf.float32) 
        self.action_upper_bound = tf.constant(abound[1].reshape(1,-1),dtype=tf.float32)
        

        self.a = self._build_a(self.S,)
        q = self._build_c(self.S, self.a, )
        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Critic')
        ema = tf.train.ExponentialMovingAverage(decay=1 - self.TAU)          # soft replacement

        def ema_getter(getter, name, *args, **kwargs):
            return ema.average(getter(name, *args, **kwargs))

        target_update = [ema.apply(a_params), ema.apply(c_params)]      # soft update operation
        a_ = self._build_a(self.S_, reuse=True, custom_getter=ema_getter)   # replaced target parameters
        q_ = self._build_c(self.S_, a_, reuse=True, custom_getter=ema_getter)

        a_loss = - tf.reduce_mean(q)  # maximize the q

        self.atrain_gradients = tf.train.AdamOptimizer(self.LR_A).minimize(a_loss, var_list=a_params)

        with tf.control_dependencies(target_update):    # soft replacement happened at here
            q_target = self.R + self.GAMMA * q_
            td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
            self.ctrain = tf.train.AdamOptimizer(self.LR_C).minimize(td_error, var_list=c_params)

        self.sess.run(tf.global_variables_initializer())

    def choose_action(self, state):
        return self.sess.run(self.a, {self.S: state})[0]


    def learn(self,batch_samples):

        batch_states = np.asarray([e[0] for e in batch_samples]).reshape(-1,self.s_dim)
        batch_actions = np.asarray([e[1] for e in batch_samples]).reshape(-1,self.a_dim)
        batch_rewards = np.asarray([e[2] for e in batch_samples]).reshape(-1,1)
        batch_next_states = np.asarray([e[3] for e in batch_samples]).reshape(-1,self.s_dim)
        
        self.sess.run(self.atrain_gradients, {self.S: batch_states})
        self.sess.run(self.ctrain, {self.S: batch_states, self.a: batch_actions, self.R: batch_rewards, self.S_: batch_next_states})

    def offline_learn(self,batch_states,batch_actions,batch_rewards,batch_next_states):
        self.sess.run(self.atrain_gradients, {self.S: batch_states})
        self.sess.run(self.ctrain, {self.S: batch_states, self.a: batch_actions, self.R: batch_rewards, self.S_: batch_next_states})
        
    def _build_a(self, state, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('Actor',reuse=reuse, custom_getter=custom_getter):
            net = state
            for i,hidden_no in enumerate([128,128,128]):
                net = tf.layers.dense(inputs=net, units=hidden_no, activation=tf.nn.tanh,name='actor_hidden'+str(i),trainable=trainable)
            sigmoid_action = tf.layers.dense(net, units=self.a_dim, name='action',activation=tf.nn.sigmoid,trainable=trainable)
            minmax_gap = self.action_upper_bound - self.action_lower_bound  
            action = minmax_gap * sigmoid_action  + self.action_lower_bound 
        return action
    
    def _build_c(self, state,action, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('Critic',reuse=reuse, custom_getter=custom_getter):
            net = tf.concat([state,action],axis=1)
            for i,hidden_no in enumerate([128,128,128]):
                net = tf.layers.dense(net,units=hidden_no,activation=tf.nn.tanh,name='critic_hidden'+str(i),trainable=trainable)
            q =  tf.layers.dense(net, units=1, name='q_value',trainable=trainable)  # Q(s,a)
        return q

    

