from __future__ import print_function

import numpy as np


#####schulman code

class LinearBaseline(object):
    def __init__(self, sess, env, name, params, train=True):
        self._coeffs = None
        self._reg_coeff = 1e-5

    def _features(self, times, observations):
        o = np.clip(observations, -10, 10)
        l = len(times)
        al = np.arange(l).reshape(-1, 1) / 100.0
        return np.concatenate([o, o ** 2, al, al ** 2, al ** 3, np.ones((l, 1))], axis=1)

    def fit(self, returns, observations):
        featmat = np.concatenate([self._features(r, o) for r, o in zip(returns, observations)])
        returns = np.concatenate([r for r in returns])
        reg_coeff = self._reg_coeff
        for _ in range(5):
            self._coeffs = np.linalg.lstsq(
                featmat.T.dot(featmat) + reg_coeff * np.identity(featmat.shape[1]),
                featmat.T.dot(returns)
            )[0]
            if not np.any(np.isnan(self._coeffs)):
                break
            reg_coeff *= 10

    def predict(self, times, observations):
        if self._coeffs is None:
            return np.zeros(len(times))
        return self._features(times, observations).dot(self._coeffs)
