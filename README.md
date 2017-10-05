# PG Agents: Policy Gradient Algorithms with Tensorflow

The idea behind pg_agents is to provide an easy to understand python package containing the state the art policy gradient algorithms.

## Implemented algorithms

- __VPG: Vanilla Policy Gradient__ Also known as REINFORCE

- __TNPG: Truncated Natural Policy Gradient__ Reformulation of the batch RL problem in terms of a contrained optimization problem

- __TRPO: Trust Region Policy Optimization__ Extension of TNPG to ensure robustness 

- __GAE: Generalized Advantage Estimator__ Method to estimate the advantage function from experience. Helps to reduce the variance of the gradient estimator.

- __PPO: Proximal Policy Optimization__ Simple but efficient extension of VPG.
