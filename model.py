import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import math

def one_hot(num_classes, class_idx):
    vector = torch.zeros((num_classes))
    vector[class_idx] = 5e2 #1e3 for eyebrow, 5e2 for blinking
    return vector

class NeuralNet(nn.Module):
    def __init__(self, num_features):
        super(NeuralNet, self).__init__()
        self.linear1 = nn.Linear(num_features, 256)
        torch.nn.init.constant_(self.linear1.bias, 0)
        torch.nn.init.xavier_normal_(self.linear1.weight, gain=1e-3)
        self.linear2 = nn.Linear(256, 256)
        torch.nn.init.constant_(self.linear2.bias, 0)
        torch.nn.init.xavier_normal_(self.linear2.weight, gain=1e-3)
        self.linear3 = nn.Linear(256, 1)
        torch.nn.init.constant_(self.linear3.bias, 0)
        torch.nn.init.xavier_normal_(self.linear3.weight, gain=1e-3)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = x.float()
        pred = self.relu(self.linear1(x))
        pred = self.tanh(self.linear3(pred))
        # pred = self.relu(self.linear2(pred))
        # pred = self.tanh(self.linear3(pred))
        return pred
    
class QLearner():
    #alpha 1e-4 for eyebrow movement and blinking
    def __init__(self, num_actions, num_features, epsilon = 5e-2, alpha=1e-4, eta=1e-2):
        self.num_actions = num_actions
        self.num_features = num_features
        self.network = NeuralNet(num_features+num_actions).to(torch.float)
        self.r_bar = 0
        self.epsilon = epsilon
        self.alpha = alpha
        self.eta = eta
        self.prev_state = None
        self.prev_action = None
        self.optimizer = torch.optim.SGD(self.network.parameters(), lr=alpha)

    def loss(self, x, reward, action, x_prime):
        predicted_reward = self.network(torch.cat((x, one_hot(self.num_actions, action))))
        with torch.no_grad():
            max_reward = torch.max(torch.tensor([self.network(torch.cat((x_prime, one_hot(self.num_actions, a)))) for a in range(self.num_actions)]))
        return (reward \
            - self.r_bar \
            + max_reward \
            - predicted_reward)**2

    def step(self, x, reward):
        #selects action using epsilon-greedy policy
        self.optimizer.zero_grad()
        action_values = []
        for action in range(self.num_actions):
            action_vector = one_hot(self.num_actions, action)
            state_action_vector = torch.cat((x, action_vector))
            action_val = self.network(state_action_vector)
            if action == 0:
                max_action = 0
                max_action_val = action_val
            elif action_val > max_action_val:
                max_action = action
                max_action_val = action_val
            action_values.append(action_val)
        rand = random.random()
        if rand < self.epsilon:
            nonmax_actions = list(range(self.num_actions))
            selected_action = random.choice(nonmax_actions)
        else:
            selected_action = max_action
        #calculate td error and backpropagate
        if (self.prev_state != None): 
            delta = self.loss(self.prev_state, reward, self.prev_action, x) # x here is S'
            self.r_bar += self.eta*self.alpha*delta.detach()
            delta.backward()
            self.optimizer.step()
        self.prev_state = x # set S
        self.prev_action = selected_action # set A
            
        return selected_action, action_values
            
