#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 11:05:35 2017

@author: daniel
"""

import numpy as np
import tensorflow as tf
import tensorflow.contrib.distributions as ds
import tensorflow.contrib.bayesflow as bf

class GaussPolicy(object):
    def __init__(self,sess,env,name,params,train=True):
        self.params=params
        self.sess=sess
        self.name=name
        self.env=env
        self.input_shape=[None ,self.params["obssize"]] #add to hyperparamters
        with tf.name_scope(self.name):
            self.input_placeholder = tf.placeholder(tf.float32,shape=self.input_shape,name="input_plh")
            self.reward_placeholder = tf.placeholder(tf.float32,shape=[None,],name="reward_plh")
            self.return_placeholder = tf.placeholder(tf.float32,shape=[None,],name="returnplh")
            self.action_placeholder = tf.placeholder(tf.float32,shape=[None,self.params["actionsize"]],name="action_plh")
            
            # self.input_shape=[None, kl_samples ,self.params["obssize"]] #add to hyperparamters
            # self.kl_input_placeholder = tf.placeholder(tf.float32,shape=self.input_shape,name="kl_input_plh")
            # self.kl_action_placeholder = tf.placeholder(tf.float32,shape=[None,kl_samples,self.params["actionsize"]],name="kl_action_plh")
            # self.kl_return_placeholder = tf.placeholder(tf.float32,shape=[None,kl_samples,],name="kl_returnplh")
        
        self.train=train
        
        self.params_list=[]
        self.params_shapes=[]

        self.buildLowDimNet()

        for p in self.params_list:
            self.params_shapes.append(p.shape.as_list())
# 
        if self.params["actionsize"]==1:
            self.dist=ds.Normal(loc=self.scaled_out,scale=tf.exp(self.log_std),allow_nan_stats=False)
        elif self.params["actionsize"] > 1:
            self.dist=ds.MultivariateNormalDiag(loc=self.scaled_out,scale_diag=tf.exp(self.log_std),allow_nan_stats=False)
            # self.dist=ds.Normal(loc=self.scaled_out,scale=tf.exp(self.log_std),allow_nan_stats=False)
            self.co=self.dist.covariance()
        else:
            raise ValueError("Invalid action dimension")

        print("Initializing with action dim",self.params["actionsize"])

        self.sa=self.dist.sample()
        self.means=self.dist.mean()
        self.std=self.dist.stddev()

        self.init_surr_loss()

    def init_surr_loss(self):
        self.lp=self.dist.log_prob(self.action_placeholder)

        self.lgrad=tf.gradients(self.lp, self.params_list)

        print(self.lp.shape)
        self.surr_loss=tf.reduce_mean(self.lp*self.return_placeholder)

        self.surr_gradient = tf.gradients(self.surr_loss, self.params_list)


    def assignParametersOp(self,params):
        op_list=[]
        for pold,pnew in zip(self.params_list,params):
            op_list.append(pold.assign(pnew))

        return op_list
      
    def buildLowDimNet(self):
        input_layer = self.input_placeholder
        
        with tf.name_scope(self.name):
            with tf.name_scope('fc1'):
                self.W_fc1 = self._weight_variable([self.params["obssize"], 64],"W_fc1",vals=(-1./np.sqrt(self.params["obssize"]),1./np.sqrt(self.params["obssize"])))
                # self.W_fc1 = self._weight_variable([self.params["obssize"], 30],"W_fc1",vals=(-0.01,np.sqrt(1/np.float(10*self.params["obssize"]))))
                self.params_list.append(self.W_fc1)
                
                self.b_fc1 = self._bias_variable([64],"b_fc1",vals=(-0.01,0.0001))
                # self.b_fc1 = self._bias_variable([30],"b_fc1",vals=(-0.01,0.0000))
                self.params_list.append(self.b_fc1)
            
                h_fc1 = tf.nn.tanh(tf.matmul(input_layer, self.W_fc1) + self.b_fc1)


            with tf.name_scope('fc2'):
                self.W_fc2 = self._weight_variable([64, 64],"W_fc2",vals=(-0.01,np.sqrt(1/64.)))
                self.params_list.append(self.W_fc2)
                
                self.b_fc2 = self._bias_variable([64],"b_fc2",vals=(-0.01,0.0001))
                self.params_list.append(self.b_fc2)
            
                h_fc2 = tf.nn.tanh(tf.matmul(h_fc1, self.W_fc2) + self.b_fc2)

            # with tf.name_scope('fc3'):
            #     self.W_fc3 = self._weight_variable([50, 25],"W_fc3",vals=(-0.01,np.sqrt(1/50.)))
            #     self.params_list.append(self.W_fc3)
                
            #     self.b_fc3 = self._bias_variable([25],"b_fc3",vals=(-0.01,0.0001))
            #     self.params_list.append(self.b_fc3)
            
            #     h_fc3 = tf.matmul(h_fc2, self.W_fc3) + self.b_fc3
                
                
            with tf.name_scope('output'):
                # self.W_fc4 = self._weight_variable([30, self.params["actionsize"]],"W_out",vals=(-0.01,np.sqrt(1/300.)))
                # self.params_list.append(self.W_fc4)
                
                # self.b_fc4 = self._bias_variable([self.params["actionsize"]],"b_out",vals=(-0.01,0.0000))
                # self.params_list.append(self.b_fc4)

                self.W_fc4 = self._weight_variable([64, self.params["actionsize"]],"W_out",vals=(-0.01,np.sqrt(1/64.)))
                self.params_list.append(self.W_fc4)
                
                self.b_fc4 = self._bias_variable([self.params["actionsize"]],"b_out",vals=(-0.01,0.01))
                self.params_list.append(self.b_fc4)

            with tf.name_scope('std_net'):
                # self.W_fc_std = self._weight_variable([self.params["obssize"], self.params["actionsize"]])
                # self.W_fc_std = tf.Variable(tf.constant(0.0,shape=[self.params["obssize"], self.params["actionsize"]],dtype=tf.float32),trainable=self.train,name="W_fc_std")
                # self.params_list.append(self.W_fc_std)
                
                # self.b_fc_std = self._bias_variable([1],"b_fc_std")
                self.b_fc_std = tf.Variable(tf.constant(self.params["init_std"],shape=[self.params["actionsize"]],dtype=tf.float32),trainable=self.train,name="b_fc_std")
                # self.b_fc_std = tf.Variable(tf.eye(1,num_columns=self.params["actionsize"]),name="b_fc_std")
                self.params_list.append(self.b_fc_std)
            
                
                
            # self.out=tf.matmul(h_fc1, self.W_fc4) + self.b_fc4
            self.out=tf.matmul(h_fc2, self.W_fc4) + self.b_fc4
            self.log_std = tf.log(self.b_fc_std+0*self.out)
            # self.scaled_out=tf.multiply(float(self.params["action_bound"]),tf.nn.tanh(self.out))
            # self.scaled_out=tf.nn.tanh(self.out)
            self.scaled_out=self.out
        return self.scaled_out

    # def buildLowDimNetNew(self):
    #     input_layer = self.input_placeholder

    #     with tf.name_scope(self.name):
    #         self.b_fc_std = tf.Variable(tf.constant(self.params["init_std"],shape=[self.params["actionsize"]],dtype=tf.float32),trainable=self.train,name="b_fc_std")
    #         self.params_list.append(self.b_fc_std)
    #         self.log_std = tf.log(self.b_fc_std)

    #         # with tf.name_scope('fc1'):
    #         layer1=tf.layers.dense(input_layer,100,activation=tf.nn.tanh,name="dense1")
                

    #         # with tf.name_scope('fc2'):
    #         layer2=tf.layers.dense(layer1,50,activation=tf.nn.tanh)

    #         # with tf.name_scope('fc3'):
    #         layer3=tf.layers.dense(layer2,25)

    #         self.scaled_out=tf.layers.dense(layer3,self.params["actionsize"])

    #         print(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope=self.name))

    def getAction(self,obs):
        a = self.sess.run([self.sa],feed_dict={self.input_placeholder:np.array(obs)})
        return a

    def getSurrLoss(self,returns,observations,actions):
        l=self.sess.run(self.surr_loss,feed_dict={
            self.input_placeholder:observations,
            self.return_placeholder:returns,
            self.action_placeholder:actions})
        return l

    def getSurrLossGrad(self,returns,observations,actions):
        g=self.sess.run(self.surr_gradient ,feed_dict={
            self.input_placeholder:observations,
            self.return_placeholder:returns,
            self.action_placeholder:actions})
        return g
 

    def _weight_variable(self,shape,name=None,vals=(-0.01,0.05)):
        # initial = tf.random_uniform(shape, minval=vals[0],maxval=vals[1])
        # initial = tf.uniform_unit_scaling_initializer(factor=1.0)
        initial = tf.truncated_normal(shape, stddev=vals[1])
        return tf.Variable(initial,trainable=self.train,name=name)
        # return tf.Variable(initial(shape=shape) ,trainable=self.train,name=name)

    def _bias_variable(self,shape,name=None,vals=(-0.01,0.05)):
        # initial = tf.random_uniform(shape, minval=vals[0],maxval=vals[1])
        initial = tf.truncated_normal(shape, stddev=vals[1])
        return tf.Variable(initial,trainable=self.train,name=name)

    def _conv2d(self,x, W, s):
        return tf.nn.conv2d(x, W, strides=[1, s, s, 1], padding='VALID')