import tensorflow as tf
import numpy as np

import pg_agents.utils.utils as utils

from pg_agents.algos.batchalgo import BatchAlgo


class TNPG(BatchAlgo):

    def compute_step_len(self, x, b, hv_prod):
        prod = np.abs(1 / np.dot(b, hv_prod(b)))
        l = np.sqrt(self.params["kl_penalty"] * prod)

        return l

    def compute_hessian_vector(self, v_init, returns, observations, actions):
        v_nested = utils.unflatten_array_list(v_init, self.policy.params_shapes)
        hv = self.sess.run(self.hvp, feed_dict=self.get_feed_dict(v_nested, returns, observations, actions))
        return utils.flatten_array_list(hv)

    def compute_natural_gradient(self, returns, observations, actions):
        b_nested = self.get_loss_grad(returns, observations, actions)
        b = utils.flatten_array_list(b_nested)
        assert np.isfinite(b).any(), "[CG] policy gradient not finite"

        x_init = [np.zeros(s) for s in self.policy.params_shapes]
        x_ = utils.flatten_array_list(x_init)

        def hv_prod(x):
            return self.compute_hessian_vector(x, returns, observations, actions)

        assert np.isfinite(hv_prod(x_)).any(), "[CG] initial product not finite"

        x_flat = utils.cg_solve(hv_prod, b, x_)

        step_len = self.compute_step_len(x_flat, b, hv_prod)
        x_out = utils.unflatten_array_list(x_flat, self.policy.params_shapes)

        return x_out, step_len

    def update_policy(self, returns, observations, actions):
        step, step_len = self.compute_natural_gradient(returns, observations, actions)
        step_len = -step_len

        fd = {plh_key: s for plh_key, s in zip(self.grad_plh, step)}
        fd[self.lr_plh] = step_len

        self.sess.run(self.apply_grads, feed_dict=fd)

        self.sess.run(self.param_assign, feed_dict={plh_key: p_entry.eval() for plh_key, p_entry in
                                                    zip(self.p_plh, self.policy.params_list)})
