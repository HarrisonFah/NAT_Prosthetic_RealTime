import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import math

class NeuralNet(nn.Module):
    def __init__(self, num_features):
        super(NeuralNet, self).__init__()
        
        #Initializes layer weights close to 0
        self.linear1 = nn.Linear(num_features, 256)
        torch.nn.init.constant_(self.linear1.bias, 0)
        torch.nn.init.xavier_normal_(self.linear1.weight, gain=1e-3)

        self.linear2 = nn.Linear(256, 1)
        torch.nn.init.constant_(self.linear2.bias, 0)
        torch.nn.init.xavier_normal_(self.linear2.weight, gain=1e-3)

        self.elu = nn.ELU()
        self.tanh = nn.Tanh() #Uses tanh as final activation to bound prediction between -1 and 1

    # Calculates prediction for average reward
    # x: State-action vector
    def forward(self, x):
        x = x.float()
        pred = self.elu(self.linear1(x))
        pred = self.tanh(self.linear2(pred))
        return pred
    
class QLearner():

    # num_actions: The total number of discrete actions the model can select from
    # num_features: Number of features describing a state
    # epsilon: Value of epsilon used in an epsilon-greedy policy (takes a random action with probability of epsilon)
    # alpha: Learning rate of model
    # eta: How much the average reward is updated by each step
    # one_hot_value: Value of selected action in one hot vector
    # filename: Prefix of save file name
    # save_freq: Saves the model every save_freq timesteps
    def __init__(self, num_actions, num_features, epsilon=5e-2, alpha=1e-4, eta=1e-2, one_hot_value=1e3, filename=None, save_freq=100):
        self.num_actions = num_actions
        self.num_features = num_features
        self.network = NeuralNet(num_features+num_actions).to(torch.float)
        self.r_bar = 0 #Average reward
        self.epsilon = epsilon
        self.alpha = alpha
        self.eta = eta
        self.one_hot_value = one_hot_value
        self.prev_state = None
        self.prev_action = None
        self.optimizer = torch.optim.SGD(self.network.parameters(), lr=alpha)
        self.filename = filename
        self.save_freq = save_freq
        self.timestep = 0

    #Returns a vector with a 1 in class_idx and 0 in all other indices
    def one_hot(self, num_classes, class_idx):
        vector = torch.zeros((num_classes))
        vector[class_idx] = self.one_hot_value
        return vector

    # Calculates the continual learning loss for a q network as defined on page 38 in this paper:
    # https://arxiv.org/abs/2006.16318
    # Note: The algorithm in this paper only states the gradient of the loss
    #
    # x: The current state
    # reward: The observed reward
    # action: The action taken
    # x_prime: The next state observed
    def loss(self, x, reward, action, x_prime):
        predicted_reward = self.network(torch.cat((x, self.one_hot(self.num_actions, action))))
        with torch.no_grad():
            max_reward = torch.max(torch.tensor([self.network(torch.cat((x_prime, self.one_hot(self.num_actions, a)))) for a in range(self.num_actions)]))
        return (reward \
            - self.r_bar \
            + max_reward \
            - predicted_reward)**2

    # Performs a single training step
    def step(self, x, reward):
        self.optimizer.zero_grad()

        #Selects an action using epsilon greedy
        action_values = []
        for action in range(self.num_actions):
            #Select the action with the maximum predicted reward
            action_vector = self.one_hot(self.num_actions, action)
            state_action_vector = torch.cat((x, action_vector))
            action_val = self.network(state_action_vector)
            if action == 0:
                max_action = 0
                max_action_val = action_val
            elif action_val > max_action_val:
                max_action = action
                max_action_val = action_val
            action_values.append(action_val)
        #Select either the maximum action with probability 1-epsilon, or a random one with probability epsilon
        rand = random.random()
        if rand < self.epsilon:
            nonmax_actions = list(range(self.num_actions))
            selected_action = random.choice(nonmax_actions)
        else:
            selected_action = max_action

        #Calculate loss and backpropagate
        if (self.prev_state != None): 
            delta = self.loss(self.prev_state, reward, self.prev_action, x) #x here is S'
            self.r_bar += self.eta*self.alpha*delta.detach() #Update the average reward
            delta.backward()
            self.optimizer.step()
        
        #Save state and action to use in next update
        self.prev_state = x # set S
        self.prev_action = selected_action # set A

        if self.filename and self.timestep % self.save_freq == 0:
            self.save(self.filename)

        self.timestep += 1
            
        return selected_action, action_values

    # Gets an action from the policy without training the model
    def get_action(self):
        #Selects an action using epsilon greedy
        action_values = []
        for action in range(self.num_actions):
            #Select the action with the maximum predicted reward
            action_vector = self.one_hot(self.num_actions, action)
            state_action_vector = torch.cat((x, action_vector))
            action_val = self.network(state_action_vector)
            if action == 0:
                max_action = 0
                max_action_val = action_val
            elif action_val > max_action_val:
                max_action = action
                max_action_val = action_val
            action_values.append(action_val)
        return max_action, action_values

    def save(self, filename):
        torch.save(self.network.state_dict(), filename + "_network")
        torch.save(self.optimizer.state_dict(), filename + "_optimizer")

    def load(self, filename):
        self.network.load_state_dict(torch.load(filename + "_network"))
        self.optimizer.load_state_dict(torch.load(filename + "_optimizer"))