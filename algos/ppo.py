import tensorflow as tf

from pg_agents.algos.vpg import VPGMiniBatch
from pg_agents.utils.kl_div import kl_div


class PPO(VPGMiniBatch):

    def init_loss(self):
        self.kl_div = tf.reduce_mean(kl_div(self.policy_old, self.policy, self.actiondim))

        rat = tf.exp(self.policy.lp-self.policy_old.lp)
        rat_clipped = tf.clip_by_value(rat, 1.-self.params["eps"], 1.+self.params["eps"])
        self.loss = tf.reduce_mean(tf.minimum(rat*self.return_placeholder, rat_clipped*self.return_placeholder))