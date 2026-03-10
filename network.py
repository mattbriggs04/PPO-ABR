import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

# actor-critic network
# mathematically we have policy = pi_theta(a_t | s_t)
# => model input is the state vector
# => model output is action (selecting a bitrate / quality level)
# also need to learn the value function V(s)
# => output both as separate heads
# could make two separate models and train them both
# but I decided to test out this method
class ACNet(nn.Module):
    def __init__(self, obs_dim, act_dim, crit_dim=1, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_act = nn.Linear(hidden_dim, act_dim)
        self.fc_crit = nn.Linear(hidden_dim, crit_dim)

    def forward(self, x):
        # x is the observation (state)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # return logits of each head actor, critic
        return self.fc_act(x), self.fc_crit(x)
