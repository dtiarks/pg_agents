#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 11:04:47 2017

@author: daniel
"""

from __future__ import print_function

import gym
from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.normalized_env import normalize
import numpy as np
import tensorflow as tf
import tensorflow.contrib.distributions as ds
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

import gaussian_policy
import linear_baseline



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
#        sess.graph.finalize()

    def initTraining(self):
        # self.kl_div=tf.reduce_mean(ds.kl_divergence(self.policy.dist,self.policy_old.dist))
        # self.hessian=tf.hessians(self.kl_div,self.policy.action_placeholder)

        self.std_op=tf.log(self.policy.dist.stddev())
        self.global_step = tf.Variable(0, trainable=False)

        self.optimizer = tf.train.AdamOptimizer(self.params['learningrate'])
        self.train=self.optimizer.apply_gradients(zip(self.policy.surr_gradient , self.policy.params_list))

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
        

        l,std=self.sess.run([self.train,self.std_op],feed_dict={
            self.policy.input_placeholder:observations,
            self.policy.action_placeholder:actions,
            self.policy.return_placeholder:returns
            })
        # print(np.mean(std))
        # kl=self.sess.run([self.hessian],feed_dict={
        #     self.policy.input_placeholder:observations,
        #     self.policy.action_placeholder:actions,
        #     self.policy.return_placeholder:returns,
        #     self.policy_old.input_placeholder:observations,
        #     self.policy_old.action_placeholder:actions,
        #     self.policy_old.return_placeholder:returns
        #     })
        # print(kl)
        # print(rewards[:,0].shape)
        return l 
    
        

# def run():
    
parser = argparse.ArgumentParser()
parser.add_argument("-E","--env", type=str, help="Mujoco task in Gym, (default: InvertedPendulum-v1)",default='MountainCarContinuous-v0')
parser.add_argument("-B","--baseline", type=bool, help="Use linear base line",default=True)
parser.add_argument("-d","--dir", type=str, help="Directory where the relevant training info is stored")
parser.add_argument("-e","--eval", type=str, help="Evaluation directory. Movies are stored here.")
parser.add_argument("-c","--checkpoint",type=str, help="Directory of latest checkpoint.")
args = parser.parse_args()
    
envname=args.env
env = gym.make(envname)
# env = normalize(CartpoleEnv())

params={
        "Env":'Pendulum-v0',
        "timesteps":500,#10000,
        "trajectories":100,
        "iterations":100,
        "discount":0.99,
        "learningrate":0.01,
        "init_std":0.99,
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
    
    baseline=linear_baseline.LinearBaseline(sess,env,"linear_baseline",params)
    vpga=VPGAgent(sess,env,params)
    
    
    np.save(os.path.join(vpga.traindir,'params_dict.npy'), params)
    epoche_name=os.path.join(vpga.traindir,"epoche_stats.tsv")

    # env = wrappers.Monitor(env, os.path.join(vpga.traindir,'monitor'), video_callable=lambda x:x%params["store_video"]==0)
    
    env.frame_skip=1
    
    rp_dtype=params["frame_dtype"]

    fshape=params["obssize"]



    newtensor_plh=tf.placeholder(tf.float32,shape=[params["obssize"],100],name="input_plh")
    T1_assign=vpga.policy.params_list[0].assign(newtensor_plh)

        
    c=0
    epoche_done=False
    rs=[]
    for e in range(params['iterations']):
        print("Starting iteration {}".format(e))
        
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
        for i in range(params['trajectories']):
            video=False
            
            if i%params["store_video"]==0:
                video=True
                # os.mkdir(os.path.join(vpga.traindir,'monitor_%d_%d'%(e,i)))
                # video_recorder = gym.monitoring.video_recorder.VideoRecorder(env=env, base_path=('monitor%d'%i), enabled=True)
            
            obsNew = env.reset()
            obsNew=np.array(obsNew).ravel()
            
            rewards=[]
            base_obs=[]
            base_times=[]
            base_actions=[]
            ret=0
            done=False
            for t in range(params['timesteps']):
                action = vpga.takeAction(obsNew)

                action=np.array(action).ravel()
                obs, r, done, _ = env.step(action)

                obs=np.array(obs).ravel()
                # if i==0:
                #     env.render()

                if done:
                    break
                
                rewards.append(r)
                
                base_times.append(t)
                base_obs.append(obsNew)
                base_actions.append(action)

                obsNew=obs
                    
                c+=1
                rcum+=r
                ret+=r
                k+=1

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


        if params["use_baseline"]:
            baseline.fit(returns, observations)
        
        print("\r[Iter: {} || Reward: {} || Frame: {} || Steps: {}]".format(i,rcum/(i+1),c,t))
        
        vpga.assignMetrics(e,feed_dict={vpga.undiscounted_return_plh:rcum/(i+1),
                                      vpga.maxreturn_plh:np.max(traj_returns)})

        print("Updating policy...")
        obs_batch=np.concatenate([o for o in observations])
        action_batch=np.concatenate([a for a in actions])
        returns_batch=np.concatenate([r for r in returns])
        advantages_batch=np.concatenate([a for a in advantages])
        if params["use_baseline"]:
            vpga.updatePolicy(advantages_batch,obs_batch,action_batch)
        else:
            vpga.updatePolicy(returns_batch,obs_batch,action_batch)


    obsNew = env.reset()
    obsNew=np.array(obsNew).ravel()

    for t in range(100*params['timesteps']):
        done=False
                    
        action = vpga.takeAction(obsNew)

        action=np.array(action).ravel()
        obs, r, done, _ = env.step(action)
        env.render("human")

        obs=np.array(obs).ravel()
        obsNew=obs

        if done:
            break



    # num=10
    # ret_t=0

    # rets_arr=np.zeros((num,num))

    # p1_arr=np.zeros((num,num))
    # p2_arr=np.zeros((num,num))

    # grad1_arr=np.zeros((num,num))
    # grad2_arr=np.zeros((num,num))

    # T1=sess.run(vpga.policy.params_list[0])

    # param_range1=np.linspace(T1[0,5]-0.8*T1[0,5],T1[0,5]+0.8*T1[0,5],num)
    # param_range2=np.linspace(T1[0,6]-0.8*T1[0,6],T1[0,6]+0.8*T1[0,6],num)

    # for i,p1 in np.ndenumerate(param_range1):
    #     for j,p2 in np.ndenumerate(param_range2):
    #         print((i,j))

    #         p1_arr[i,j]=p1
    #         p2_arr[i,j]=p2

    #         T1=sess.run(vpga.policy.params_list[0])
    #         T1[0,5]=p1
    #         T1[0,6]=p2

    #         sess.run(T1_assign,feed_dict={newtensor_plh:T1})


    #         returns=[]#np.empty((params['trajectories']*params['timesteps']))
    #         returns_mean=[]#np.empty((params['trajectories']*params['timesteps']))
    #         observations=[]#np.empty((params['trajectories']*params['timesteps'],params["obssize"]))
    #         actions=[]#np.empty((params['trajectories']*params['timesteps'],params["actionsize"]))

    #         traj_returns=[]
            
    #         t1=time.clock()
    #         k=0
    #         rcum=0
    #         for _ in range(params['trajectories']):
    #             obsNew = env.reset()
    #             obsNew=np.array(obsNew).ravel()
                
    #             rewards=[]
    #             ret=0
    #             for t in range(100*params['timesteps']):
    #                 done=False
                    
    #                 action = vpga.takeAction(obsNew)

    #                 action=np.array(action).ravel()
    #                 obs, r, done, _ = env.step(action)

    #                 obs=np.array(obs).ravel()


    #                 if done:
    #                     break
                    
    #                 rewards.append(r)
    #                 observations.append(obs)
    #                 actions.append(action)

    #                 obsNew=obs
                        
    #                 c+=1
    #                 rcum+=r
    #                 ret+=r
    #                 k+=1

    #             traj_returns.append(ret)

    #             return_disc=0
    #             returns_l=[]
    #             for tx in range(len(rewards)-1,-1,-1):
    #                 return_disc=rewards[tx]+params['discount']*return_disc
    #                 returns_l.append(return_disc)
    #             returns_l=returns_l[::-1]

    #             for rx in returns_l:
    #                 returns.append(np.array(rx))

    #             returns_mean.append(returns_l[0])

    #         rets_arr[i,j]=np.mean(returns_mean)

    #         returns=np.expand_dims(returns, axis=1)
    #         grads=vpga.policy.getSurrLossGrad(np.array(returns),np.array(observations),np.array(actions))
    #         grad1_arr[i,j]=grads[0][0][5]
    #         grad2_arr[i,j]=grads[0][0][6]


    # plt.subplot(211)
    # plt.imshow(rets_arr, interpolation='bicubic')

    # plt.subplot(212)
    # plt.quiver(p1_arr,p2_arr,grad1_arr,grad2_arr,rets_arr)
    # plt.show()

    
    env.close()

# if __name__ == '__main__':      
#     run()