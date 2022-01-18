import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1, fc2, fc3, fc4, fc5, layer_agent):
        """Initialize parameters and build model.
        """
        
        self.layer_agent = layer_agent
        
        if self.layer_agent == 1:
            # 5 Layers
            super(Actor, self).__init__()
            self.seed = torch.manual_seed(seed)
            self.bn0 = nn.BatchNorm1d(state_size)
            self.fc1 = nn.Linear(state_size, fc1)
            self.bn1 = nn.BatchNorm1d(fc1)
            self.fc2 = nn.Linear(fc1, fc2)
            self.bn2 = nn.BatchNorm1d(fc2)
            self.fc3 = nn.Linear(fc2, fc3)
            self.bn3 = nn.BatchNorm1d(fc3)
            self.fc4 = nn.Linear(fc3, fc4)
            self.bn4 = nn.BatchNorm1d(fc4)         
            self.fc5 = nn.Linear(fc5, action_size)     
            self.reset_parameters()
        else:
            # 3 Layers
            super(Actor, self).__init__()
            self.seed = torch.manual_seed(seed)
            self.bn0 = nn.BatchNorm1d(state_size)
            self.fc1 = nn.Linear(state_size, fc1)
            self.bn1 = nn.BatchNorm1d(fc1)
            self.fc2 = nn.Linear(fc1, fc2)
            self.bn2 = nn.BatchNorm1d(fc2)
            self.fc3 = nn.Linear(fc3, action_size)     
            self.reset_parameters()           

    def reset_parameters(self):
        if self.layer_agent == 1:
            # 5 Layers
            self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
            self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
            self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
            self.fc4.weight.data.uniform_(*hidden_init(self.fc4))
            self.fc5.weight.data.uniform_(-3e-3, 3e-3)
        else:
            # 3 Layers
            self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
            self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
            self.fc3.weight.data.uniform_(-3e-3, 3e-3)            

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        if self.layer_agent == 1:
            # 5 Layers
            x = F.relu(self.fc1(state))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            x = F.relu(self.fc4(x))
            return torch.tanh(self.fc5(x))
        else:
            # 3 Layers
            x = F.relu(self.fc1(state))
            x = F.relu(self.fc2(x))
            return torch.tanh(self.fc3(x))           

class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, fc6, fc7, fc8, fc9, fc10, fc11, layer_critic):
        """Initialize parameters and build model.
        """
        
        self.layer_critic = layer_critic
        
        if self.layer_critic == 1:
            # 6 Layers
            super(Critic, self).__init__()
            self.seed = torch.manual_seed(seed)
            self.bn0 = nn.BatchNorm1d(state_size)
            self.fc6 = nn.Linear(state_size, fc6)
            self.bn1 = nn.BatchNorm1d(fc6)
            self.fc7 = nn.Linear(fc6+action_size, fc7)
            self.fc8 = nn.Linear(fc7, fc8)
            self.fc9 = nn.Linear(fc8, fc9)
            self.fc10 = nn.Linear(fc9, fc10)
            self.fc11 = nn.Linear(fc11, 1)
            self.reset_parameters()
        else:
            # 4 Layers
            super(Critic, self).__init__()
            self.seed = torch.manual_seed(seed)
            self.bn0 = nn.BatchNorm1d(state_size)
            self.fc6 = nn.Linear(state_size, fc6)
            self.bn1 = nn.BatchNorm1d(fc6)
            self.fc7 = nn.Linear(fc6+action_size, fc7)
            self.fc8 = nn.Linear(fc7, fc8)
            self.fc9 = nn.Linear(fc9, 1)
            self.reset_parameters()           

    def reset_parameters(self):
        if self.layer_critic == 1:
            # 6 Layers
            self.fc6.weight.data.uniform_(*hidden_init(self.fc6))
            self.fc7.weight.data.uniform_(*hidden_init(self.fc7))
            self.fc8.weight.data.uniform_(*hidden_init(self.fc8))
            self.fc9.weight.data.uniform_(*hidden_init(self.fc9))
            self.fc10.weight.data.uniform_(*hidden_init(self.fc10))
            self.fc11.weight.data.uniform_(-3e-3, 3e-3)
        else:
            # 4 Layers
            self.fc6.weight.data.uniform_(*hidden_init(self.fc6))
            self.fc7.weight.data.uniform_(*hidden_init(self.fc7))
            self.fc8.weight.data.uniform_(*hidden_init(self.fc8))
            self.fc9.weight.data.uniform_(-3e-3, 3e-3)            
            
    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        if self.layer_critic == 1:
            # 6 Layers
            xs = F.leaky_relu(self.fc6(state))
            x = torch.cat((xs, action), dim=1)
            x = F.leaky_relu(self.fc7(x))
            x = F.leaky_relu(self.fc8(x))
            x = F.leaky_relu(self.fc9(x))
            x = F.leaky_relu(self.fc10(x))
            return self.fc11(x)
        else:
            # 4 Layers
            xs = F.leaky_relu(self.fc6(state))
            x = torch.cat((xs, action), dim=1)
            x = F.leaky_relu(self.fc7(x))
            x = F.leaky_relu(self.fc8(x))
            return self.fc9(x)            