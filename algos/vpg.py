import numpy as np

from pg_agents.algos.batchalgo import BatchAlgo


class VPG(BatchAlgo):
    def update_policy(self, returns, observations, actions):
        fd = {self.lr_plh: -self.params['learningrate'],
              self.input_placeholder: observations,
              self.action_placeholder: actions,
              self.return_placeholder: returns}

        self.sess.run(self.minimize, feed_dict=fd)
        self.sess.run(self.param_assign, feed_dict={plh_key: p_entry.eval() for plh_key, p_entry in
                                                    zip(self.p_plh, self.policy.params_list)})


class VPGMiniBatch(BatchAlgo):
    def _get_next_batch(self, returns, observations, actions, size):
        idx = np.arange(len(returns))
        batch_idx = np.random.choice(idx, size)
        mask = np.array(np.ones(len(returns)), dtype=np.bool)
        mask[batch_idx] = False

        m_returns = returns[batch_idx]
        m_observations = observations[batch_idx, ...]
        m_actions = actions[batch_idx, ...]

        returns = returns[mask]
        observations = observations[mask, ...]
        actions = actions[mask, ...]

        return m_returns, m_observations, m_actions, returns, observations, actions

    # def update_policy(self, returns, observations, actions):
    #     _returns, _observations, _actions = returns.copy(), observations.copy(), actions.copy()
    #
    #     for i in range(self.params["epochs"]):
    #         m_returns, m_observations, m_actions, _returns, _observations, _actions = self._get_next_batch(_returns, _observations, _actions, self.params["mini_batch_size"])
    #         fd = {self.lr_plh: -self.params['learningrate'],
    #               self.input_placeholder: m_observations,
    #               self.action_placeholder: m_actions,
    #               self.return_placeholder: m_returns}
    #         _, l = self.sess.run([self.minimize, self.loss], feed_dict=fd)
    #
    #     # print("Batch loss:", l)
    #     self.sess.run(self.param_assign, feed_dict={plh_key: p_entry.eval() for plh_key, p_entry in
    #                                                 zip(self.p_plh, self.policy.params_list)})

    def update_policy(self, returns, observations, actions):
        index = range(len(returns))

        for i in range(self.params["epochs"]):
            idx = np.random.choice(index, size=self.params["mini_batch_size"])
            fd = {self.lr_plh: -self.params['learningrate'],
                  self.input_placeholder: observations[idx, ...],
                  self.action_placeholder: actions[idx, ...],
                  self.return_placeholder: returns[idx, ...]}
            _, l = self.sess.run([self.minimize, self.loss], feed_dict=fd)

        print("Batch loss:", l)
        self.sess.run(self.param_assign, feed_dict={plh_key: p_entry.eval() for plh_key, p_entry in
                                                    zip(self.p_plh, self.policy.params_list)})
