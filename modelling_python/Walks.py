
# imports 
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
from scipy.optimize import minimize
import os
import sys

"""

Generate walks depending on the environment you choose


"""

        

class MatchingLaw(): #hard coded for 2 arms 

    def __init__(self, k, n, step_size, hazard_rate, lb, hb) -> None:
        
        
        self.k = k #k-arms
        self.n = n #n-trials
        self.step_size = step_size 
        self.hazard_rate = hazard_rate # probability of step
        self.lb = lb# lower bound = min value of step 
        self.hb = hb # upper bound = max value of step
        
    def generate_walk(self, plot, plt_title):

        # array of walks for k-arms and n-trials 
        walk = np.zeros((self.k, self.n)) 

        #random initial state at time_step=0 
        while True:
            walk[:, 0] = np.round(np.random.dirichlet(np.ones(self.k)), decimals=1)*100
            if np.sum(walk[:, 0]) == 100: 
                break

        #in each trial
        for i in range(1, self.n): 
            
            #First randomly select an arm
            j = np.random.choice(range(self.k),self.k, replace=False)

            # check if step happens
            if np.random.binomial(1, self.hazard_rate):
                
                #if step, increase or decrease
                if np.random.binomial(1, 0.5) ==1: 
                    
                    #increase 
                    walk[j[0], i] =  walk[j[0], i-1] + self.step_size
                    walk[j[1], i] =  100-walk[j[0], i] 
                    
                    #keep probabilities within bounds
                    if walk[j[0],i] > self.hb: 
                        walk[j[0],i]=self.hb
                        walk[j[1],i]=100-walk[j[0], i] 
                    
                else:   
                    #decrease
                    walk[j[0], i] =  walk[j[0], i-1] - self.step_size
                    walk[j[1], i] =  100- walk[j[0], i] 

                    #keep probabilities within bounds
                    if walk[j[0],i] < self.lb: 
                        walk[j[0],i]=self.lb
                        walk[j[1],i]=100-walk[j[0], i] 
                          
        
            # if no step, keep the same
            else:
                walk[j, i] = walk[j, i-1]
    
        #Normalize to 1 (sum to 100 because of weird glitches when using floats)
        walk = walk/100 
        
        if np.sum(walk[:, -1]) != 1.0:
            raise ValueError("The walk does not sum to 1.0")        

        
        fig = None

        if plot:
            # plot the walk
            fig, ax = plt.subplots()
            col= ['tab:red','tab:blue'] #define colour for each arm 
            colors = iter(col)

            for i in range(self.k):
                c = next(colors)
                ax.plot(np.arange(self.n), walk[i], color=c)
            ax.set_xlabel('Trial #')
            ax.set_ylabel('Reward probability')
            ax.set_title(plt_title)

        return walk, fig






class KArmedBandit():

    def __init__(self, k, n, step_size, hazard_rate, lb, hb) -> None:
        
        self.k = k
        
        self.n = n
        self.step_size = step_size
        self.hazard_rate = hazard_rate
        self.lb = lb
        self.hb = hb
        
    def generate_walk(self, plot, plt_title):
        walk = np.zeros((self.k, self.n))
        walk[:, 0] = np.random.choice(np.arange(0.1, 1, 0.1), size=self.k)
        
        for i in range(1, self.n):

            # TODO: vectorize iteration over k?
            for j in range(self.k):
                
                # check if step happens
                if np.random.binomial(1, self.hazard_rate):

                    # check if step increases or decreases
                    if np.random.binomial(1, 0.5):
                        walk[j, i] = walk[j, i-1] + self.step_size
                    else:
                        walk[j, i] = walk[j, i-1] - self.step_size

                # no change in prob
                else:
                    walk[j, i] = walk[j, i-1]
        
                # keep probabilities within the bound
                if walk[j, i] > self.hb:
                    walk[j, i] = self.hb

                elif walk[j, i] < self.lb:
                    walk[j, i] = self.lb
        
        fig = None
        if plot:
            # plot the walk
            fig, ax = plt.subplots()
            col= ['tab:red','tab:blue'] #define colour for each arm 
            colors = iter(col)
            for i in range(self.k):
                c=next(colors)
                ax.plot(np.arange(self.n), walk[i], color=c)
            ax.set_xlabel('Trial #')
            ax.set_ylabel('Reward probability')
            ax.set_title(plt_title)

        return walk, fig