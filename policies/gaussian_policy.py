#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 11:05:35 2017

@author: daniel
"""

import numpy as np
import tensorflow as tf
import tensorflow.contrib.distributions as ds


class GaussPolicy(object):
    def __init__(self, sess, env, name, input_plh, action_plh, params, train=True, std_val=None, actiondim=1):
        self.params = params
        self.sess = sess
        self.name = name
        self.env = env
        self.adapt_std = True
        self.actiondim = actiondim

        self.train = train

        self.params_list = []
        self.params_shapes = []

        if std_val is not None:
            std = std_val
            self.adapt_std = False
            self.build_low_dim_net(input_plh)
        else:
            self.build_low_dim_net(input_plh)
            std = tf.exp(self.log_std)

        for p in self.params_list:
            self.params_shapes.append(p.shape.as_list())

        if self.actiondim == 1:
            self.dist = ds.Normal(loc=self.scaled_out, scale=std, allow_nan_stats=False)
        elif self.actiondim > 1:
            self.dist = ds.MultivariateNormalDiag(loc=self.scaled_out, scale_diag=std, allow_nan_stats=False)
            self.co = self.dist.covariance()
        else:
            raise ValueError("Invalid action dimension")

        print("Initializing gaussian policy with action dim", self.actiondim)

        self.sa = self.dist.sample()
        self.means = self.dist.mean()
        self.std = self.dist.stddev()
        self.lp = self.dist.log_prob(action_plh)
        self.prob = self.dist.prob(action_plh)

    def assign_parameters_op(self, params):
        op_list = []
        for pold, pnew in zip(self.params_list, params):
            op_list.append(pold.assign(pnew))

        return op_list

    def build_low_dim_net(self, input_layer):

        with tf.name_scope(self.name):
            with tf.name_scope('fc1'):
                self.W_fc1 = self._weight_variable([self.params["obssize"], 64], "W_fc1", vals=(
                    -1. / np.sqrt(self.params["obssize"]), 1. / np.sqrt(self.params["obssize"])))
                self.params_list.append(self.W_fc1)

                self.b_fc1 = self._bias_variable([64], "b_fc1", vals=(-0.01, 0.0001))
                self.params_list.append(self.b_fc1)

                h_fc1 = tf.nn.tanh(tf.matmul(input_layer, self.W_fc1) + self.b_fc1)

            with tf.name_scope('fc2'):
                self.W_fc2 = self._weight_variable([64, 64], "W_fc2", vals=(-0.01, np.sqrt(1 / 64.)))
                self.params_list.append(self.W_fc2)

                self.b_fc2 = self._bias_variable([64], "b_fc2", vals=(-0.01, 0.0001))
                self.params_list.append(self.b_fc2)

                h_fc2 = tf.nn.tanh(tf.matmul(h_fc1, self.W_fc2) + self.b_fc2)

            with tf.name_scope('output'):
                self.W_fc4 = self._weight_variable([64, self.actiondim], "W_out",
                                                   vals=(-0.01, np.sqrt(1 / 64.)))
                self.params_list.append(self.W_fc4)

                self.b_fc4 = self._bias_variable([self.actiondim], "b_out", vals=(-0.01, 0.01))
                self.params_list.append(self.b_fc4)

            self.out = tf.matmul(h_fc2, self.W_fc4) + self.b_fc4

            if self.adapt_std is not None:
                with tf.name_scope('std_net'):
                    self.b_fc_std = tf.Variable(
                        tf.constant(self.params["init_std"], shape=[self.actiondim], dtype=tf.float32),
                        trainable=self.train, name="b_fc_std")
                    self.params_list.append(self.b_fc_std)
                    self.log_std = tf.log(self.b_fc_std + 0 * self.out)

            self.scaled_out = self.out

    def _weight_variable(self, shape, name=None, vals=(-0.01, 0.05)):
        initial = tf.truncated_normal(shape, stddev=vals[1])
        return tf.Variable(initial, trainable=self.train, name=name)

    def _bias_variable(self, shape, name=None, vals=(-0.01, 0.05)):
        initial = tf.truncated_normal(shape, stddev=vals[1])
        return tf.Variable(initial, trainable=self.train, name=name)
