#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 11:04:47 2017

@author: daniel
"""

from __future__ import print_function

import gym
from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.box2d.double_pendulum_env import DoublePendulumEnv
from rllab.envs.normalized_env import normalize
import numpy as np
import scipy as sp
import tensorflow as tf
import tensorflow.contrib.distributions as ds
from tensorflow.python.ops.gradients_impl import _hessian_vector_product
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
import utils

import pybullet_envs
import pybullet_envs.bullet.racecarGymEnv as re
from pybullet_envs.bullet.kukaGymEnv import KukaGymEnv

from tensorflow.python import debug as tf_debug

import gaussian_policy
import linear_baseline

def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)

class VPGAgent(object):
    
    def __init__(self,sess,env,params):
        self.params=params
        self.cnt=0
        self.env=env
        self.sess=sess
        self.current_loss=0
        
#        self.run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
#        self.run_metadata = tf.RunMetadata()
        
        
        self.policy=gaussian_policy.GaussPolicy(sess,env,"gaussian_policy",params)
        self.policy_old=gaussian_policy.GaussPolicy(sess,env,"gaussian_policy_old",params,train=False)

        
        self.initTraining()
        self.initSummaries()
        
#        os.mkdir(self.params['traindir'])
        subdir=datetime.datetime.now().strftime('%d%m%y_%H%M%S')
        self.traindir=os.path.join(params['traindir'], "run_%s"%subdir)
        os.mkdir(self.traindir)
        self.picdir=os.path.join(self.traindir,"pics")
        os.mkdir(self.picdir)
        checkpoint_dir=os.path.join(self.traindir,self.params['checkpoint_dir'])
        os.mkdir(checkpoint_dir)
        
        self.saver = tf.train.Saver()
        
        if params["latest_run"]:
            self.latest_traindir=os.path.join(params['traindir'], "run_%s"%params["latest_run"])
            latest_checkpoint = tf.train.latest_checkpoint(os.path.join(self.latest_traindir,self.params['checkpoint_dir']))
            if latest_checkpoint:
                print("Loading model checkpoint {}...\n".format(latest_checkpoint))
                self.saver.restore(sess, latest_checkpoint)
        
        self.merged = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(self.traindir,sess.graph)
                
        init = tf.global_variables_initializer()
        
        sess.run(init)

        sess.run(self.param_assign,feed_dict={plh_key:p_entry.eval() for plh_key,p_entry in zip(self.p_plh,self.policy.params_list)})
        
        sess.graph.finalize()


    def _hv_grad(self, v, kl,params):
        g1=tf.gradients(kl,params)
        g_prod=[g*v_s for g,v_s in zip(g1,v)]
        hessian_prod=tf.gradients(g_prod,params)
        return hessian_prod


    def _mvn_kl_div_full(self,a,b):
        co0=a.co
        co1=b.co
        co1_i=tf.matrix_inverse(co1)
        mu=b.means-a.means
        k=self.params["actionsize"]

        A=tf.trace(tf.matmul(co0,co1_i))
        B=tf.squeeze(tf.matmul(tf.matmul(tf.expand_dims(mu,1),co1_i),tf.expand_dims(mu,2)),axis=[1,2])
        det_co1=tf.matrix_determinant(co1)
        det_co0=tf.matrix_determinant(co0)
        C=tf.log(det_co1/det_co0)
        ret=0.5*(A+B-k+C)
        return ret

    def _mvn_kl_div_diag(self,a,b):
        co0=a.co
        co1=b.co
        co0_diag=tf.matrix_diag_part(co0)
        co1_diag=tf.matrix_diag_part(co1)
        co1_i=tf.matrix_diag(1./co1_diag)
        
        det_co1=tf.reduce_prod(co1_diag,axis=1)
        det_co0=tf.reduce_prod(co0_diag,axis=1)

        mu=b.means-a.means
        k=self.params["actionsize"]

        A=tf.trace(tf.matmul(co0,co1_i))
        B=tf.squeeze(tf.matmul(tf.matmul(tf.expand_dims(mu,1),co1_i),tf.expand_dims(mu,2)),axis=[1,2])
        
        C=tf.log(det_co1/det_co0)
        ret=0.5*(A+B-k+C)
        return ret

    def _mvn_kl_div(self,a,b):
        return self._mvn_kl_div_diag(a,b)

    def initTraining(self):
        #into policy
        self.p_plh=[
            tf.placeholder(tf.float32,shape=s,name="assign_plh_%d"%i) for i,s in zip(range(len(self.policy.params_shapes)),self.policy.params_shapes)
            ]
        self.param_assign=self.policy_old.assignParametersOp(self.p_plh)
        self.param_assign_new=self.policy.assignParametersOp(self.p_plh)
        
        #stays in algo (only kl div algos)
        self.kl_div=tf.reduce_mean(self._mvn_kl_div(self.policy_old,self.policy))

        # self.g1=tf.gradients(self.kl_test,self.policy.params_list)

        #stays in algo (own function?)
        self.v_plh=[
            tf.placeholder(tf.float32,shape=s,name="cg_vector_plh_%d"%i) for i,s in zip(range(len(self.policy.params_shapes)),self.policy.params_shapes)
            ]
        self.hv2=_hessian_vector_product(self.kl_div,self.policy.params_list, self.v_plh)

        #goes into policy
        self.grad_plh=[
            tf.placeholder(tf.float32,shape=s,name="grad_plh_%d"%i) for i,s in zip(range(len(self.policy.params_shapes)),self.policy.params_shapes)
            ]

        #????
        self.global_step = tf.Variable(0, trainable=False)

        #into policy
        self.lr_plh = tf.placeholder(tf.float32,name="learningrate_plh")
        self.optimizer = tf.train.AdamOptimizer(self.lr_plh)
        self.apply_grads=self.optimizer.apply_gradients(zip(self.grad_plh , self.policy.params_list))

    #more here
    def initSummaries(self):
        self.undiscounted_return=tf.Variable(0,name="undiscounted_return",dtype=tf.float32,trainable=False)
        self.undiscounted_return_plh=tf.placeholder(tf.float32,name="undiscounted_return_plh")
        self.ur_assign_op=self.undiscounted_return.assign(self.undiscounted_return_plh)

        self.maxreturn=tf.Variable(0,name="maxreturn",dtype=tf.float32,trainable=False)
        self.maxreturn_plh=tf.placeholder(tf.float32,name="maxreturn_plh")
        self.maxret_op=self.maxreturn.assign(self.maxreturn_plh)
        
        with tf.name_scope("stats"):
            tf.summary.scalar('return', self.undiscounted_return)
            tf.summary.scalar('maxreturn', self.maxreturn)

    def assignMetrics(self, step, feed_dict):
        _,_,summary=self.sess.run([self.ur_assign_op,self.maxret_op,self.merged],feed_dict=feed_dict)

        self.train_writer.add_summary(summary, step)
    
    def takeAction(self,obs):
        o=np.expand_dims(obs, axis=0)
        a=self.policy.getAction(o)
        return a

    def updatePolicy(self,returns,observations,actions):
        v=[np.random.random(s) for s in self.policy.params_shapes]
        fd2={plh_key:v_entry for plh_key,v_entry in zip(self.v_plh,v)}
        fd2[self.policy.input_placeholder]=observations
        fd2[self.policy.action_placeholder]=actions
        fd2[self.policy.return_placeholder]=returns
        fd2[self.policy_old.input_placeholder]=observations
        fd2[self.policy_old.action_placeholder]=actions
        fd2[self.policy_old.return_placeholder]=returns
        
        step,step_len=self.computeNaturalGradient(returns,observations,actions)

        params_backup=[p.eval() for p in self.policy.params_list]
        step_len=-self.params["init_step"]
        
        for i in range(200):
            fd={plh_key:s for plh_key,s in zip(self.grad_plh,step)}
            fd[self.lr_plh]=step_len

            self.sess.run(self.apply_grads,feed_dict=fd)
            
            kl=self.sess.run(self.kl_div,feed_dict=fd2)

            # print("kl",kl)
            # print("kl_diag",kl_diag)
            # print("step_len",step_len)
            if kl<self.params["kl_penalty"]:
                break

            step_len=step_len/self.params["beta"]
            self.sess.run(self.param_assign_new,feed_dict={plh_key:p_entry for plh_key,p_entry in zip(self.p_plh,params_backup)})
        
        # step=self.policy.getSurrLossGrad(returns,observations,actions)
        # step_len=-0.01
        # fd={plh_key:s for plh_key,s in zip(self.grad_plh,step)}
        # fd[self.lr_plh]=step_len

        # self.sess.run(self.apply_grads,feed_dict=fd)
            
        self.sess.run(self.param_assign,feed_dict={plh_key:p_entry.eval() for plh_key,p_entry in zip(self.p_plh,self.policy.params_list)})
        

    #goes into TNPG class
    def buidFeedDict(self,v,returns,observations,actions):
        fd={plh_key:v_entry for plh_key,v_entry in zip(self.v_plh,v)}
        fd[self.policy.input_placeholder]=observations
        fd[self.policy.action_placeholder]=actions
        fd[self.policy.return_placeholder]=returns
        fd[self.policy_old.input_placeholder]=observations
        fd[self.policy_old.action_placeholder]=actions
        fd[self.policy_old.return_placeholder]=returns

        return fd

    def computeHessianVector(self,v_init,returns,observations,actions):
        v_nested=utils.unflatten_array_list(v_init,self.policy.params_shapes)
        hv=self.sess.run(self.hv2,feed_dict=self.buidFeedDict(v_nested,returns,observations,actions))
        return utils.flatten_array_list(hv)

    def computeNaturalGradient(self,returns,observations,actions):
        b_nested=self.policy.getSurrLossGrad(returns,observations,actions)
        b=utils.flatten_array_list(b_nested)
        assert np.isfinite(b).any(), "[CG] policy gradient not finite"

        x_init=[np.zeros(s) for s in self.policy.params_shapes]
        x_=utils.flatten_array_list(x_init)


        Ax_prod=lambda x:self.computeHessianVector(x,returns,observations,actions)
        assert np.isfinite(Ax_prod(x_)).any(), "[CG] initial product not finite"
        
        x_flat=utils.cg_solve(Ax_prod,b,x_)

        # prod=np.abs(1/np.dot(b,Ax_prod(b)))# TNPG step size
        prod=np.abs(2/np.dot(x_flat,Ax_prod(x_flat)))# TRPO initial step size
        step_len=np.sqrt(self.params["kl_penalty"]*prod)
        x_out=utils.unflatten_array_list(x_flat,self.policy.params_shapes)

        return x_out,step_len
            
        
        

# def run():
    
parser = argparse.ArgumentParser()
parser.add_argument("-E","--env", type=str, help="Mujoco task in Gym, (default: InvertedPendulum-v1)",default='MountainCarContinuous-v0')
parser.add_argument("-B","--baseline", type=bool, help="Use linear base line",default=False)
parser.add_argument("-d","--dir", type=str, help="Directory where the relevant training info is stored")
parser.add_argument("-e","--eval", type=str, help="Evaluation directory. Movies are stored here.")
parser.add_argument("-c","--checkpoint",type=str, help="Directory of latest checkpoint.")
args = parser.parse_args()
    
envname=args.env
env = gym.make(envname)
# env = re.RacecarGymEnv(isDiscrete=False ,renders=True)
# env = KukaGymEnv(renders=True)

params={
        "Env":'Pendulum-v0',
        "timesteps":2000,#10000,
        "trajectories":100,
        "iterations":600,
        "discount":0.99,
        "learningrate":0.01,
        "init_std":1.,
        "init_step":0.05,
        "kl_penalty":0.1,
        "beta":1.5,
        "batch_size":50000,
        "use_baseline":args.baseline,
        "actionsize": env.action_space.shape[0],
        "obssize": env.observation_space.shape[0],
        "traindir":"./train_dir",
        "summary_steps":100,
        "skip_episodes": 50,
        "framewrite_episodes":100,
        "checkpoint_dir":'checkpoints',
        "checkpoint_steps":200000,
        "latest_run":args.checkpoint,
        "metricupdate":10,
        "frame_shape":(64,64,3),
        "frame_dtype":np.float32,
        "high_dim":False,
        "action_bound":env.action_space.high[0],
        "store_video":50
}

params["Env"]=envname

tf.reset_default_graph()

with tf.Session() as sess:
    # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    # sess.add_tensor_filter("infs", tf_debug.has_inf_or_nan)

    
    baseline=linear_baseline.LinearBaseline(sess,env,"linear_baseline",params)
    vpga=VPGAgent(sess,env,params)
    
    
    np.save(os.path.join(vpga.traindir,'params_dict.npy'), params)
    epoche_name=os.path.join(vpga.traindir,"epoche_stats.tsv")

    
    rp_dtype=params["frame_dtype"]

    fshape=params["obssize"]

    overall_return=[]
        
    c=0
    epoche_done=False
    rs=[]
    for e in range(params['iterations']):
        paths = []
        returns=[]#np.empty((params['trajectories']*params['timesteps']))
        observations=[]#np.empty((params['trajectories']*params['timesteps'],params["obssize"]))
        actions=[]#np.empty((params['trajectories']*params['timesteps'],params["actionsize"]))
        advantages=[]
        times=[]

        traj_returns=[]
        
        t1=time.clock()
        k=0
        rcum=0
        for i in range(1000*params['trajectories']):
            obsNew = env.reset()
            obsNew=np.array(obsNew).ravel()
            
            rewards=[]
            base_obs=[]
            base_times=[]
            base_actions=[]
            ret=0
            done=False
            spin_once=False
            for t in range(params['timesteps']):
                
                action = vpga.takeAction(obsNew)
                action=np.array(action).ravel()

                obs, r, done, _ = env.step(action)

                obs=np.array(obs).ravel()
                # if i==0:
                #     env.render()
                # env.render()
                if done:
                    break
                
                spin_once=True
                
                rewards.append(r)
                base_times.append(t)
                base_obs.append(obsNew)
                base_actions.append(action)

                obsNew=obs
                    
                c+=1
                rcum+=r
                ret+=r
                k+=1


            if spin_once:
                traj_returns.append(ret)
                observations.append(np.array(base_obs))
                actions.append(np.array(base_actions))
                times.append(base_times)

                if params["use_baseline"]:
                    bs=baseline.predict(base_times,base_obs)
                
                return_disc=0
                returns_l=[]
                adv_l=[]
                for tx in range(len(rewards)-1,-1,-1):
                    return_disc=rewards[tx]+params['discount']*return_disc
                    returns_l.append(return_disc)
                    if params["use_baseline"]:
                        adv_l.append(return_disc-bs[tx])
                returns_l=np.array(returns_l[::-1])
                adv_l=np.array(adv_l[::-1])
                
                adv_l = (adv_l - np.mean(adv_l)) / (np.std(adv_l) + 1e-8)

                returns.append(returns_l)
                advantages.append(adv_l)

            
            if k>params['batch_size']:
                print("k",k)
                break


        if params["use_baseline"]:
            baseline.fit(returns, observations)
        
        overall_return.append(rcum/(i+1))
        print("\r[Iter: {} || Reward: {} || Frame: {} || Steps: {}]".format(e,rcum/(i+1),c,t))
        
        vpga.assignMetrics(e,feed_dict={vpga.undiscounted_return_plh:rcum/(i+1),
                                      vpga.maxreturn_plh:np.max(traj_returns)})

        obs_batch=np.concatenate([o for o in observations])
        action_batch=np.concatenate([a for a in actions])
        returns_batch=np.concatenate([r for r in returns])
        advantages_batch=np.concatenate([a for a in advantages])
        batch_len=obs_batch.shape[0]
        batch_idx=np.arange(batch_len)
        
        t1=time.clock()
        if params["use_baseline"]:
            vpga.updatePolicy(advantages_batch,obs_batch,action_batch)
        else:
            vpga.updatePolicy(returns_batch,obs_batch,action_batch)
        t2=time.clock()
        dt=(t2-t1)
        print("update time",dt)

    obsNew = env.reset()
    obsNew=np.array(obsNew).ravel()
    video_recorder = gym.monitoring.video_recorder.VideoRecorder(env=env, base_path=(os.path.join(vpga.traindir,envname)), enabled=True)

    for t in range(50*params['timesteps']):
        done=False
                        
        action = vpga.takeAction(obsNew)

        action=np.array(action).ravel()
        obs, r, done, _ = env.step(action)
        # env.render("human")
        video_recorder.capture_frame()

        obs=np.array(obs).ravel()
        obsNew=obs

        if done:
            break
        
    video_recorder.close()
    env.close()

# if __name__ == '__main__':      
#     run()