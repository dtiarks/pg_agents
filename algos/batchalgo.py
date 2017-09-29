import tensorflow as tf
from tensorflow.python.ops.gradients_impl import _hessian_vector_product

import numpy as np
import datetime
import os

from pg_agents.utils.kl_div import kl_div
from pg_agents.policies.gaussian_policy import GaussPolicy


class BatchAlgo(object):
    # TODO: Add checkpoint saver
    def __init__(self, sess, env, params):
        self.params = params
        self.cnt = 0
        self.env = env
        self.sess = sess
        self.current_loss = 0
        self.input_shape = [None, self.params["obssize"]]

        #        self.run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        #        self.run_metadata = tf.RunMetadata()

        self.init_training()
        self.init_summaries()

        #        os.mkdir(self.params['traindir'])
        subdir = datetime.datetime.now().strftime('%d%m%y_%H%M%S')
        self.traindir = os.path.join(params['traindir'], "run_%s" % subdir)
        os.mkdir(self.traindir)
        checkpoint_dir = os.path.join(self.traindir, self.params['checkpoint_dir'])
        os.mkdir(checkpoint_dir)

        self.saver = tf.train.Saver()

        if params["latest_run"]:
            self.latest_traindir = os.path.join(params['traindir'], "run_%s" % params["latest_run"])
            latest_checkpoint = tf.train.latest_checkpoint(
                os.path.join(self.latest_traindir, self.params['checkpoint_dir']))
            if latest_checkpoint:
                print("Loading model checkpoint {}...\n".format(latest_checkpoint))
                self.saver.restore(sess, latest_checkpoint)

        self.merged = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(self.traindir, sess.graph)

        init = tf.global_variables_initializer()

        sess.run(init)

        sess.run(self.param_assign,
                 feed_dict={plh_key: p_entry.eval() for plh_key, p_entry in zip(self.p_plh, self.policy.params_list)})

    def init_placeholders(self):
        self.input_placeholder = tf.placeholder(tf.float32, shape=self.input_shape, name="input_plh")
        self.reward_placeholder = tf.placeholder(tf.float32, shape=[None, ], name="reward_plh")
        self.return_placeholder = tf.placeholder(tf.float32, shape=[None, ], name="returnplh")
        self.action_placeholder = tf.placeholder(tf.float32, shape=[None, self.params["actionsize"]], name="action_plh")

    def init_policy(self):
        self.policy = GaussPolicy(self.sess, self.env, "gaussian_policy", self.input_placeholder,
                                  self.action_placeholder, self.params)
        self.policy_old = GaussPolicy(self.sess, self.env, "gaussian_policy_old", self.input_placeholder,
                                      self.action_placeholder, self.params,
                                      train=False)

    def init_loss(self):
        self.kl_div = tf.reduce_mean(kl_div(self.policy_old, self.policy, self.params["actionsize"]))

        self.loss = tf.reduce_mean(self.policy.lp * self.return_placeholder)

    def init_gradients(self):
        self.loss_gradient = tf.gradients(self.loss, self.policy.params_list)
        self.hvp = _hessian_vector_product(self.kl_div, self.policy.params_list, self.v_plh)

        self.grad_plh = [
            tf.placeholder(tf.float32, shape=s, name="grad_plh_%d" % i) for i, s in
            zip(range(len(self.policy.params_shapes)), self.policy.params_shapes)
        ]

    def init_training(self):
        self.init_placeholders()
        self.init_policy()
        self.init_loss()
        self.p_plh = [
            tf.placeholder(tf.float32, shape=s, name="assign_plh_%d" % i) for i, s in
            zip(range(len(self.policy.params_shapes)), self.policy.params_shapes)
        ]
        self.param_assign = self.policy_old.assignParametersOp(self.p_plh)
        self.param_assign_new = self.policy.assignParametersOp(self.p_plh)

        self.v_plh = [
            tf.placeholder(tf.float32, shape=s, name="cg_vector_plh_%d" % i) for i, s in
            zip(range(len(self.policy.params_shapes)), self.policy.params_shapes)
        ]
        self.init_gradients()

        self.lr_plh = tf.placeholder(tf.float32, name="learningrate_plh")
        self.optimizer = tf.train.AdamOptimizer(self.lr_plh)
        self.apply_grads = self.optimizer.apply_gradients(zip(self.grad_plh, self.policy.params_list))
        self.minimize = self.optimizer.minimize(self.loss, var_list=self.policy.params_list)

    def init_summaries(self):
        self.undiscounted_return = tf.Variable(0, name="undiscounted_return", dtype=tf.float32, trainable=False)
        self.undiscounted_return_plh = tf.placeholder(tf.float32, name="undiscounted_return_plh")
        self.ur_assign_op = self.undiscounted_return.assign(self.undiscounted_return_plh)

        self.maxreturn = tf.Variable(0, name="maxreturn", dtype=tf.float32, trainable=False)
        self.maxreturn_plh = tf.placeholder(tf.float32, name="maxreturn_plh")
        self.maxret_op = self.maxreturn.assign(self.maxreturn_plh)

        with tf.name_scope("stats"):
            tf.summary.scalar('return', self.undiscounted_return)
            tf.summary.scalar('maxreturn', self.maxreturn)

    def assign_metrics(self, step, feed_dict):
        _, _, summary = self.sess.run([self.ur_assign_op, self.maxret_op, self.merged], feed_dict=feed_dict)

        self.train_writer.add_summary(summary, step)

    def get_action(self, obs):
        o = np.expand_dims(obs, axis=0)
        a = self.sess.run(self.policy.sa, feed_dict={self.input_placeholder: o})
        return a

    def get_mean(self, obs):
        # o = np.expand_dims(obs, axis=0)
        m = self.sess.run(self.policy.means, feed_dict={self.input_placeholder: obs})
        return m

    def get_loss(self, returns, observations, actions):
        l = self.sess.run(self.loss, feed_dict={
            self.input_placeholder: observations,
            self.return_placeholder: returns,
            self.action_placeholder: actions})
        return l

    def get_loss_grad(self, returns, observations, actions):
        g = self.sess.run(self.loss_gradient, feed_dict={
            self.input_placeholder: observations,
            self.return_placeholder: returns,
            self.action_placeholder: actions})
        return g

    def get_feed_dict(self, v, returns, observations, actions):
        fd = {plh_key: v_entry for plh_key, v_entry in zip(self.v_plh, v)}
        fd[self.input_placeholder] = observations
        fd[self.action_placeholder] = actions
        fd[self.return_placeholder] = returns

        return fd

    def update_policy(self, returns, observations, actions):
        raise NotImplementedError
