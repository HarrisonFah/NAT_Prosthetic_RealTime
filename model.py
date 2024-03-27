import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import math

def one_hot(num_classes, class_idx):
    vector = torch.zeros((num_classes))
    vector[class_idx] = 1
    return vector

class NeuralNet(nn.Module):
    def __init__(self, num_features):
        super(NeuralNet, self).__init__()
        self.linear1 = nn.Linear(num_features, num_features, bias=False)
        torch.nn.init.zeros_(self.linear1.weight)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(num_features, 1, bias=False)
        torch.nn.init.zeros_(self.linear2.weight)

    def forward(self, x):
        x = x.float()
        pred = self.linear1(x)
        pred = self.relu(pred)
        pred = self.linear2(pred)
        return pred
    
class QLearner():
    def __init__(self, num_actions, num_features, epsilon = 5e-2, alpha=1e-1, eta=1e-3):
        self.num_actions = num_actions
        self.num_features = num_features
        self.network = NeuralNet(num_features+num_actions).to(torch.float)
        print(self.network)
        self.r_bar = 0
        self.epsilon = epsilon
        self.alpha = alpha
        self.eta = eta
        self.optimizer = torch.optim.SGD(self.network.parameters(), lr=alpha)

    def loss(self, x, reward, action):
        print("reward:", reward)
        print("self.r_bar:", self.r_bar)
        print("max Q(S', a):", torch.max(torch.tensor([self.network(torch.cat((x, one_hot(self.num_actions, a)))) for a in range(self.num_actions)])))
        print("Q(S,A):", self.network(torch.cat((x, one_hot(self.num_actions, action)))))
        return reward \
            - self.r_bar \
            + torch.max(torch.tensor([self.network(torch.cat((x, one_hot(self.num_actions, a)))) for a in range(self.num_actions)])) \
            - self.network(torch.cat((x, one_hot(self.num_actions, action))))

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
            nonmax_actions.remove(max_action)
            selected_action = random.choice(nonmax_actions)
        else:
            selected_action = max_action
        #calculate td error and backpropagate
        predicted_reward = self.network(torch.cat((x, one_hot(self.num_actions, action))))
        delta = self.loss(x, reward, selected_action)
        print("loss:", delta)
        self.r_bar += self.eta*self.alpha*delta.detach()
        delta.backward()
        self.optimizer.step()
        return selected_action, predicted_reward
            
