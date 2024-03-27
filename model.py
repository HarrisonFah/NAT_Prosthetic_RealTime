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
        torch.nn.init.kaiming_uniform_(self.linear1.weight, nonlinearity="relu")
        #torch.nn.init.zeros_(self.linear1.bias)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(num_features, 1, bias=False)
        torch.nn.init.kaiming_uniform_(self.linear2.weight, nonlinearity="relu")
        #torch.nn.init.zeros_(self.linear2.bias)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = x.float()
        pred = self.linear1(x)
        pred = self.relu(pred)
        pred = self.linear2(pred)
        pred = self.sig(pred)

        return pred
    
class QLearner():
    def __init__(self, num_actions, num_features, epsilon = 5e-2, alpha=1e-3, eta=1e-3):
        self.num_actions = num_actions
        self.num_features = num_features
        self.network = NeuralNet(num_features+num_actions).to(torch.float)
        print(self.network)
        self.r_bar = 0
        self.epsilon = epsilon
        self.alpha = alpha
        self.eta = eta
        self.curr_state = None
        self.curr_reward = None
        self.curr_action = None
        self.optimizer = torch.optim.SGD(self.network.parameters(), lr=alpha)

    def loss(self, x, reward, action, x_prime):
        print("reward:", reward)
        print("self.r_bar:", self.r_bar)
        print("max Q(S', a):", torch.max(torch.tensor([self.network(torch.cat((x_prime, one_hot(self.num_actions, a)))) for a in range(self.num_actions)])))
        print("Q(S,A):", self.network(torch.cat((x, one_hot(self.num_actions, action)))))
        return reward \
            - self.r_bar \
            + torch.max(torch.tensor([self.network(torch.cat((x_prime, one_hot(self.num_actions, a)))) for a in range(self.num_actions)])) \
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
            selected_action = random.choice(nonmax_actions)
        else:
            selected_action = max_action
        #calculate td error and backpropagate
        predicted_reward = self.network(torch.cat((x, one_hot(self.num_actions, action))))
        if (self.curr_state != None): 
            delta = self.loss(self.curr_state, reward, self.curr_action, x) # x here is S'
            print(delta.detach())
            self.r_bar += self.eta*self.alpha*delta.detach()
            delta.backward()
            print(self.network.linear1.weight.grad)
            print(self.network.linear2.weight.grad)
            self.optimizer.step()
        self.curr_state = x # set S
        #self.curr_reward = predicted_reward  # set R
        self.curr_action = selected_action # set A
            
        return selected_action, predicted_reward
            
