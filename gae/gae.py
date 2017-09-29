import numpy as np
import tensorflow as tf

from pg_agents.algos.batchalgo import BatchAlgo
from pg_agents.algos.vpg import VPGMiniBatch
from pg_agents.utils.kl_div import kl_div


class GAE(VPGMiniBatch):

    def init_loss(self):
        self.kl_div = tf.reduce_mean(kl_div(self.policy_old, self.policy, self.params["actionsize"]))

        self.loss = -1.*tf.losses.mean_squared_error(self.return_placeholder, self.policy.scaled_out[:,0])
