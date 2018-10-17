
# Tennis project report.

In this paper we are going to do an analysis of the [Deep Deterministic Policy Gradient](https://arxiv.org/abs/1509.02971) algorithm in the Tennis environment of [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents) with the competition of 2 agents playing tennis.

To follow this project you can execute the python notebook [Tennis.ipynb](Tennis.ipynb). 
The first cell of the notebook is to set the environment plus some functions to make the code easier.
In the second code cell is where the agents are trained to learn the task acording with a certain hyperparameters.
The third and last cell is to see how the agents behave once trained.

The agents (2) are implemented in [Agents.py](Agents.py). This class depends on [ddpg.py](ddpg.py) and  [model.py](model.py). The first define one single agent following [DDPG algorithm](https://arxiv.org/abs/1509.02971) and the second define the network chosen for an agent.
It is considered that the agents have learned when they get a +0.5 combined reward for 100 episodes.

## 1.- First steps

I started out with the DDPG agent from a [previous project](https://github.com/gregoriomezquita/ml-agents/tree/master/Reacher).
Actor network consists of 3 fully connected layers with Relu activations and a final Tanh non-linear output.
```
Actor(
  (model): Sequential(
    (0): BatchNorm1d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (1): NoisyLinear (24 -> 32, Factorized: True)
    (2): ReLU()
    (3): BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (4): NoisyLinear (32 -> 32, Factorized: True)
    (5): ReLU()
    (6): BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (7): NoisyLinear (32 -> 2, Factorized: True)
    (8): Tanh()
  )
)
```
The Critic has also 3 fully connected layers with Relu activations.
```
Critic(
  (model_input): Sequential(
    (0): Linear(in_features=24, out_features=128, bias=True)
    (1): ReLU()
    (2): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
  (model_output): Sequential(
    (0): Linear(in_features=130, out_features=128, bias=True)
    (1): ReLU()
    (2): Linear(in_features=128, out_features=1, bias=True)
  )
)
```

The following hyperparameters are the starting point:
```
config= {
    "actor_lr": 1e-3,
    "critic_lr": 1e-3,
    "actor_nodes": [32, 32],
    "critic_nodes": [128, 128],
    "batch_size": 256,
    "memory_size": 100000,
    "discount": 0.9,
    "tau": 0.001,
    "action_noise": "No",    # Options: No, Normal, OU, 
    "sigma": 0.1,            # OUNoise, Normal
    "critic_l2_reg": 0.0,  # 1e-2
}
```

## 2.- Hyperparameters selection

## 3.- Solution

<p align="center">
  <img src="images/Tennis-sigma-0.01-actor-64.png">
</p>

<p align="center">
  <img src="images/Tennis-first-success-node-128-sigma-0.1.gif">
</p>


## 4.- Conclusions

## 5.- Improvements
