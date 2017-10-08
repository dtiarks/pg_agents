import numpy as np
import tensorflow as tf
import argparse
import time
import os
import sys
import pylab as plt
import importlib

import gym
import roboschool

# import pybullet_envs
# import pybullet_envs.bullet.racecarGymEnv as re
# from pybullet_envs.bullet.kukaGymEnv import KukaGymEnv

sys.path.append('/'.join(str.split(__file__, '/')[:-2]))

from pg_agents.algos.batchalgo import BatchAlgo
from pg_agents.algos.vpg import VPG, VPGMiniBatch
from pg_agents.algos.tnpg import TNPG
from pg_agents.algos.trpo import TRPO
from pg_agents.algos.ppo import PPO
from pg_agents.gae.gae import GAE
from pg_agents.gae.linear_baseline import LinearBaseline


def train_experiment(env, sess, algo, params):
    agent = algo(sess, env, params, actiondim=params["actionsize"])
    gae = GAE(sess, env, params)
    baseline = LinearBaseline(sess, env, "linear_baseline", params)

    # sess.graph.finalize()

    overall_return = []

    c = 0
    for e in range(params['iterations']):
        returns = []
        observations = []
        actions = []
        advantages = []
        times = []
        num_steps = []

        traj_returns = []
        traj_adv = []
        preds = []
        mean_disc_returns = []

        t1 = time.clock()
        k = 0
        rcum = 0
        for i in range(params['trajectories']):
            obs_new = env.reset()
            obs_new = np.array(obs_new).ravel()

            rewards = []
            base_obs = []
            base_obs2 = []
            base_times = []
            base_actions = []
            dones = []
            ret = 0
            done = False
            spin_once = False

            for t in range(params['timesteps']):
                action = agent.get_action(obs_new)
                action = np.array(action).ravel()

                obs, r, done, _ = env.step(action)

                obs = np.array(obs).ravel()
                # if i==0:
                #     env.render()
                # env.render()
                if done:
                    break

                spin_once = True

                rewards.append(r)
                base_times.append(t)
                base_obs.append(obs_new)
                base_obs2.append(obs)
                base_actions.append(action)
                dones.append(float(done))

                obs_new = obs

                c += 1
                rcum += r
                ret += r
                k += 1

            num_steps.append(t)

            if spin_once:
                traj_returns.append(ret)
                observations.append(np.array(base_obs))
                actions.append(np.array(base_actions))
                times.append(base_times)

                if params["use_linear_baseline"]:
                    bs = baseline.predict(base_times, base_obs)

                if params["use_gae"]:
                    v_gae = gae.get_mean(base_obs)[:, 0]
                    v_gae2 = gae.get_mean(base_obs2)[:, 0]
                    preds.append(v_gae[0])

                return_disc = 0
                adv = 0
                returns_l = []
                adv_l = []
                for tx in range(len(rewards) - 1, -1, -1):
                    if params["use_gae"]:
                        dt_err = rewards[tx] + params['discount'] * v_gae2[tx] * (1 - dones[tx]) - v_gae[tx]
                        adv = adv * (params['discount'] * params['lambda_gae']) + dt_err
                        adv_l.append(adv)

                    return_disc = rewards[tx] + params['discount'] * return_disc
                    returns_l.append(return_disc)

                    if params["use_linear_baseline"] and not params["use_gae"]:
                        adv_l.append(return_disc - bs[tx])

                returns_l = np.array(returns_l[::-1])
                adv_l = np.array(adv_l[::-1])

                mean_disc_returns.append(returns_l[0])
                traj_adv.append(np.mean(adv_l))

                if params["normalize_gae"]:
                    adv_l = (adv_l - np.mean(adv_l)) / (np.std(adv_l) + 1e-8)

                returns.append(returns_l)
                advantages.append(adv_l)

            if k > params['batch_size']:
                break

        overall_return.append(rcum / (i + 1))

        mean_num_steps = np.mean(num_steps)
        print(
            "\r[Iter: {} || Reward: {:.2f} || Total: {} || Steps: {:.2f}]".format(e, rcum / (i + 1), c, mean_num_steps))

        obs_batch = np.concatenate([o for o in observations])
        action_batch = np.concatenate([a for a in actions])
        returns_batch = np.concatenate([r for r in returns])
        advantages_batch = np.concatenate([a for a in advantages])

        gae_loss = gae.get_loss(returns_batch, obs_batch, np.expand_dims(returns_batch, 1))

        agent.assign_metrics(e, feed_dict={agent.undiscounted_return_plh: rcum / (i + 1),
                                           agent.maxreturn_plh: np.max(traj_returns),
                                           agent.meanadv_plh: np.mean(traj_adv),
                                           agent.loss_summ_plh: gae_loss})

        t1 = time.clock()
        if params["use_gae"]:
            agent.update_policy(advantages_batch, obs_batch, action_batch)
            gae.update_policy(returns_batch, obs_batch, np.expand_dims(returns_batch, 1))
        else:
            if params["use_linear_baseline"]:
                baseline.fit(returns, observations)
                agent.update_policy(advantages_batch, obs_batch, action_batch)
            else:
                agent.update_policy(returns_batch, obs_batch, action_batch)

        t2 = time.clock()
        dt = (t2 - t1)
        print("update time", dt)

    agent.save_check_point()
    gae.save_check_point()

    return overall_return, agent


def run_experiment(env, sess, agent, params):
    """

    :param env: the environment
    :type env: int
    :param sess:
    :param agent:
    :param params:
    """
    obs_new = env.reset()
    obs_new = np.array(obs_new).ravel()
    video_recorder = gym.monitoring.video_recorder.VideoRecorder(env=env,
                                                                 base_path=(os.path.join(agent.traindir, envname)),
                                                                 enabled=True)

    for t in range(params['timesteps']):
        action = agent.get_action(obs_new)

        action = np.array(action).ravel()
        obs, r, done, _ = env.step(action)

        video_recorder.capture_frame()

        obs = np.array(obs).ravel()
        obs_new = obs

        if done:
            break

    video_recorder.close()


def run_stored_agent(env, sess, params, render=True):
    """

    :param env: the environment
    :type env: int
    :param sess:
    :param agent:
    :param params:
    """
    agent = BatchAlgo(sess, env, params, actiondim=params["actionsize"])
    obs_new = env.reset()
    obs_new = np.array(obs_new).ravel()

    if render is not True:
        video_recorder = gym.monitoring.video_recorder.VideoRecorder(env=env,
                                                                     base_path=(os.path.join(agent.traindir, envname)),
                                                                     enabled=True)

    for t in range(params['timesteps']):
        action = agent.get_action(obs_new)

        action = np.array(action).ravel()
        obs, r, done, _ = env.step(action)

        if render:
            env.render("human")
        else:
            video_recorder.capture_frame()

        obs = np.array(obs).ravel()
        obs_new = obs

        if done:
            break

    if render is not True:
        video_recorder.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-E", "--env", type=str, help="Mujoco task in Gym, (default: InvertedPendulum-v1)",
                        default='MountainCarContinuous-v0')
    parser.add_argument("-B", "--baseline", type=bool, help="Use linear base line", default=False)
    parser.add_argument("-d", "--dir", type=str, help="Directory where the relevant training info is stored")
    parser.add_argument("-e", "--eval", type=str, help="Evaluation directory. Movies are stored here.")
    parser.add_argument("-c", "--checkpoint", type=str, help="Directory of latest checkpoint.")
    args = parser.parse_args()

    envname = args.env
    env = gym.make(envname)

    params = {
        "Env": envname,
        "timesteps": 2000,
        "trajectories": 2000,
        "iterations": 5000,
        "discount": 0.99,
        "learningrate": 3e-4,
        "adam_eps": 1e-5,
        "init_std": 0.6,
        "init_step": 0.05,
        "kl_penalty": 0.1,
        "beta": 1.75,
        "linearsearch_steps": 50,
        "eps": 0.2,
        "batch_size": 2048,
        "mini_batch_size": 64,
        "epochs": 10,
        "use_linear_baseline": False,
        "use_gae": True,
        "lambda_gae": 0.95,
        "normalize_gae": True,
        "actionsize": env.action_space.shape[0],
        "obssize": env.observation_space.shape[0],
        "traindir": "./train_dir",
        "summary_steps": 100,
        # "load_model": None,
        "load_model": "./train_dir/run_071017_192435/checkpoints/checkpoint-0",
        "checkpoint_dir": 'checkpoints',
        "checkpoint_steps": 200000,
        "metricupdate": 10,
        "action_bound": env.action_space.high[0],
    }

    tf.reset_default_graph()

    with tf.Session() as sess:
        # return1, agent1 = train_experiment(env, sess, PPO, params)
        # env.close()
        #
        # np.save(os.path.join(agent1.traindir, 'params_dict.npy'), params)

        # env = gym.make(envname)
        # run_experiment(env, sess, agent1, params)
        # env.close()

        env = gym.make(envname)
        run_stored_agent(env, sess, params)
        env.close()

    # plt.title("TRPO\nEnv: {}".format(params["Env"]))
    # plt.plot(return1, label="TRPO")
    # # plt.plot(return2, label="TRPO")
    # plt.legend()
    # plt.show()
