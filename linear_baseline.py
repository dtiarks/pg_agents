from __future__ import print_function

import gym
# from rllab.envs.box2d.cartpole_env import CartpoleEnv
# from rllab.envs.normalized_env import normalize
import numpy as np
import tensorflow as tf
import time
from collections import deque  
import datetime
import os
import sys
from gym import wrappers
import argparse
from tensorflow.python.client import timeline
import matplotlib.pyplot as plt
import roboschool

class LinearBaseline(object):
    def __init__(self,sess,env,name,params,train=True):
        self.params=params
        self.sess=sess
        self.name=name
        self.env=env
        self.input_shape=[None ,self.params["obssize"]] #add to hyperparamters
        with tf.name_scope(self.name):
            self.input_placeholder_lin = tf.placeholder(tf.float32,shape=self.input_shape,name="input_plh_lin")
            self.reward_placeholder = tf.placeholder(tf.float32,shape=[None,1],name="reward_plh_lin")
            self.return_placeholder = tf.placeholder(tf.float32,shape=[None,1],name="returnplh_lin")
            self.time_placeholder = tf.placeholder(tf.float32,shape=[None,1],name="time_plh_lin")
        
        self.train=train
        
        self.params_list=[]

        self.init_model()
        self.init_loss()

    def init_loss(self):
        self.loss = tf.reduce_mean(tf.square(self.out - self.return_placeholder))

        optimizer = tf.train.GradientDescentOptimizer(0.01)
        grads=optimizer.compute_gradients(self.loss,var_list=self.params_list)
        self.train = optimizer.apply_gradients(grads)


      
    def init_model(self):
        s = self.input_placeholder_lin
        t = self.time_placeholder/1000.

        x = tf.concat([s,tf.square(s),t,t**2,t**3,tf.ones_like(t)],axis=1)#,tf.pow((0.01*t),2),tf.pow((0.01*t),3)

        with tf.name_scope(self.name):
            with tf.name_scope('linear_input'):
                initial_W = tf.constant(-1.5,shape=(14,1))
                self.W = tf.Variable(initial_W,name="W_lin")
                self.params_list.append(self.W)
                
                initial_b = tf.constant(10.,shape=(1,))
                self.b = tf.Variable(initial_b,name="b_lin")
                self.params_list.append(self.b)
            
                
            self.out = tf.reduce_sum(tf.matmul(x,self.W) + self.b,1)
            self.out=tf.transpose(self.out)

        # return self.out

    def fit(self,returns,observations,times):
        # print(observations.shape)
        # print(times.shape)
        # returns=np.transpose(returns)
        for i in range(500):
            curr_loss=self.sess.run([self.train,self.loss], feed_dict={
                self.input_placeholder_lin:observations,
                self.time_placeholder:times,
                self.return_placeholder:returns
                })

        # curr_loss = sess.run(self.loss, feed_dict={
        #         self.input_placeholder_lin:observations,
        #         self.return_placeholder:returns,
        #         self.time_placeholder:times})
            # print("Baseline fit: loss: %d"%(curr_loss).shape))
            print(np.array(curr_loss))


    def predict(self,observations,times):
        b=self.sess.run([self.out], feed_dict={
                self.input_placeholder_lin:observations,
                self.time_placeholder:times
                })
        return b

env = gym.make('RoboschoolInvertedPendulum-v1')

params={
        "Env":'RoboschoolInvertedPendulum-v1',
        "timesteps":500,#10000,
        "trajectories":100,
        "iterations":350,
        "discount":0.99,
        "learningrate":0.002,
        "init_std":1.0,
        "actionsize": env.action_space.shape[0],
        "obssize": env.observation_space.shape[0],
        "traindir":"./train_dir",
        "summary_steps":100,
        "skip_episodes": 50,
        "framewrite_episodes":100,
        "checkpoint_dir":'checkpoints',
        "checkpoint_steps":200000,
        "metricupdate":10,
        "frame_shape":(64,64,3),
        "frame_dtype":np.float32,
        "high_dim":False,
        "action_bound":env.action_space.high[0],
        "store_video":50
}



tf.reset_default_graph()

with tf.Session() as sess:
    baseline=LinearBaseline(sess,env,'linear',params)
