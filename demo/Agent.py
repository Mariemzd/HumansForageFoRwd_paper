
"""
This module defines the Q-learning and Foraging Agents class , which interacts with bandit-like environments.

"""

import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
import plotly.graph_objects as go
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle


class QAgent():

    """ 

    Represents an Q-learning agent (comparing-value algorithm).

    Attributes:
    k (float) : number of arms
    n (float) : number of trials 
    walks (2-D array) :  the bandit walk (1st dimension is number of arms, 2nd dimension is number of trials)  
    alpha (float) : learning rate
    beta (float) : temperature variable
    epsilon (float) : not sure
  

    """

    def __init__(self,k,n, walks, alpha, beta) -> None:

        """
        Initialize the agent with the given parameters.

        """


        # walk
        
        self.k = k
        self.n = n
        self.walks = walks # walk
        
        # model parameters
        self.alpha = alpha
        self.beta = beta 
       
        
    
        # preallocation
        self.q = np.zeros(self.k) + 1/self.k # value of each arm
        self.p = np.zeros(self.k) #probability of choice
        self.last_choice = -1
        self.last_reward = 0

        # containers
        self.choice_history = []
        self.reward_history = []
        self.p_reward_diff = []
        self.p_reward_choice = []
        self.reward_history = []
        self.q_history = self.q
        self.p_history = self.p
    

    def choose_step(self, i):
        '''
        Agent makes a choice.

        '''


        if self.last_choice == -1 or np.sum(self.q)== 0.0: # first trial is ore
            oit = 0
        else:
            # agent chooses to oit or ore
            oit = 1
        
        if oit == 0: # if explore
            # randomly choose an arm 
            self.last_choice = np.random.choice(np.arange(self.k)) 

        else: # if exploit

            # softmax choosing
            self.p = np.exp(self.q*self.beta)/np.sum(np.exp(self.q*self.beta))
            self.last_choice = np.random.choice(np.arange(self.k), 1, p = self.p)

        # now find out if they got rewarded, based on what they saw
        self.last_reward = np.random.binomial(1, self.walks[self.last_choice, i])

        # value update for the next trial
        self.q[self.last_choice] += self.alpha*(self.last_reward - self.q[self.last_choice]) 

        # save some information
        self.choice_history = np.hstack([self.choice_history,self.last_choice]) # append last choice
        self.reward_history = np.hstack([self.reward_history,self.last_reward]) # append last rwd
        self.q_history = np.vstack([self.q_history,self.q])
        self.p_history = np.vstack([self.p_history,self.p])

        # calculate difference in reward btw chosen and unchosen
        p_reward_choice = self.walks[self.last_choice, i]
        p_reward_other = (np.sum(self.walks[:, i]) - self.walks[self.last_choice, i]) / (self.k-1)
        p_reward_diff = p_reward_choice - p_reward_other
        self.p_reward_diff.append(p_reward_diff)
        
        # also keep p(reward last choice)
        self.p_reward_choice.append(self.walks[self.last_choice, i])

    def walk(self):
        for i in range(self.n):
            self.choose_step(i)

    def get_choice_history(self):
        return np.array(self.choice_history)

    def get_reward_history(self):
        return np.array(self.reward_history)
    
    def get_q_history(self):
        return np.array(self.q_history)
    
    def get_p_history(self):
        return np.array(self.p_history)
    






class ForagingAgent():
    """ 

    Represents an foraging agent (compare-to-threshold algorithm).

    Attributes:

    alpha (float) : learning rate
    beta (float) : temperature variable
    rho (float) : threshold 
    walks (2-D array) :  the bandit walk (1st dimension is number of arms, 2nd dimension is number of trials)  

    """

    def __init__(self,k , n, walks, alpha, beta, rho) -> None:

        # walk
        self.k = k
        self.n = n
        self.walks= walks # walk
        
        # model parameters
        self.alpha = alpha
        self.beta = beta 
        self.rho = rho

        # preallocation
        self.v_oit = 1
        self.last_choice = -1
        self.last_reward = 0 

        # containers
        self.choice_history = []
        self.reward_history = []
        self.p_reward_diff = []
        self.p_reward_choice = []
        self.reward_history = []
        self.q_history = self.v_oit
        self.p_history = []
        
    def choose_step(self, i):
        if self.last_choice == -1: # first trial is ore
            oit = 0
            oit_prob= 0
        else:
            # otherwise, agent must draw ore/oit
            oit_prob = 1/(1+(np.exp(-1*(self.v_oit-self.rho)* self.beta))) # softmax w/ temperature
            oit = np.random.binomial(1, oit_prob)

        if oit == 0: # if they're exploring, we need to choose an option
            self.v_oit = self.rho # reset value of exploitation to 0
            # randomly choose a different arm
            other_choices = np.arange(self.k)
            # this line makes it EXCLUSIVE CHOOSING - useful b/c otherwise foraging will just say to do exactly what you did last time
            other_choices = other_choices[other_choices != self.last_choice] # thisis the exclusive line
            self.last_choice = np.random.choice(other_choices) # sample from distribution

        # now find out if they got rewarded, based on what they saw
        self.last_reward = np.random.binomial(1, self.walks[self.last_choice, i])

        # value update for the next trial
        self.v_oit += self.alpha*(self.last_reward - self.v_oit) 

        # save some information
        self.choice_history = np.hstack([self.choice_history,self.last_choice]) # append last choice
        self.reward_history = np.hstack([self.reward_history,self.last_reward]) # append last rwd
        self.q_history = np.vstack([self.q_history,self.v_oit])
        self.p_history = np.hstack([self.p_history,oit_prob]) ###MARIEM TO VERIFY

        # calculate difference in reward btw chosen and unchosen
        p_reward_choice = self.walks[self.last_choice, i]
        p_reward_other = (np.sum(self.walks[:, i]) - self.walks[self.last_choice, i]) / (self.k-1)
        p_reward_diff = p_reward_choice - p_reward_other
        self.p_reward_diff.append(p_reward_diff)
        
        # also keep p(reward last choice)
        self.p_reward_choice.append(self.walks[self.last_choice, i])

    def walk(self):
        for i in range(self.n):
            self.choose_step(i)

    def get_choice_history(self):
        return np.array(self.choice_history)

    def get_reward_history(self):
        return np.array(self.reward_history)

    def get_q_history(self):
        return np.array(self.q_history)

    def get_p_history(self):
        return np.array(self.p_history)

