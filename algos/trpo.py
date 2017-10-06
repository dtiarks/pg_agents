from builtins import print

import tensorflow as tf
import numpy as np

from pg_agents.algos.tnpg import TNPG


class TRPO(TNPG):
    def compute_step_len(self, x, b, hv_prod):
        prod = np.abs(2 / np.dot(x, hv_prod(x)))
        l = np.sqrt(self.params["kl_penalty"] * prod)

        return l

    def update_policy(self, returns, observations, actions):
        v = [np.random.random(s) for s in self.policy.params_shapes]
        fd2 = {plh_key: v_entry for plh_key, v_entry in zip(self.v_plh, v)}
        fd2[self.input_placeholder] = observations
        fd2[self.action_placeholder] = actions
        fd2[self.return_placeholder] = returns

        step, step_len = self.compute_natural_gradient(returns, observations, actions)

        params_backup = [p.eval() for p in self.policy.params_list]
        step_len = -step_len

        for i in range(self.params["linearsearch_steps"]):
            fd = {plh_key: s for plh_key, s in zip(self.grad_plh, step)}
            fd[self.lr_plh] = step_len

            self.sess.run(self.apply_grads, feed_dict=fd)

            kl = self.sess.run(self.kl_div, feed_dict=fd2)

            if kl < self.params["kl_penalty"]:
                break

            step_len = step_len / self.params["beta"]
            self.sess.run(self.param_assign_new,
                          feed_dict={plh_key: p_entry for plh_key, p_entry in zip(self.p_plh, params_backup)})

        self.sess.run(self.param_assign, feed_dict={plh_key: p_entry.eval() for plh_key, p_entry in
                                                    zip(self.p_plh, self.policy.params_list)})
