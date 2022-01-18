from scipy.spatial.distance import euclidean
from scipy.spatial.distance import cityblock

import random
import numpy as np
import pandas as pd

class environment:
        
    def row_values(self,df,t,state_list,num_agents,state_size):
        
        state = df.iloc[t][state_list].values.flatten().tolist()
        num_rows = num_agents
        num_columns = state_size
        
        states = np.zeros(shape=(num_rows,num_columns))
    
        for row in range(num_agents):
            states[row] = state 
            
        return states    
    
    def step(self,done,reward,actions,price,BuyPrice,Budget,InitialBudget):
        
        #print("actions: ", actions)       
        
        # Select action
        action = actions.tolist()[0]
        action = action.index(max(action))
        # Hold
        if action == 0:
            pass
        # Buy    
        elif action == 1 and Budget >= price and BuyPrice == 0:
            reward = 0.00000000001
            BuyPrice = price
            Budget -= price
        # Sell         
        elif action == 2 and BuyPrice != 0:
            reward = price/BuyPrice - 1
            BuyPrice = 0
            Budget += price
        # Incorrect Action
        else:
            reward = - 1
        # Game Over 
        if Budget <= 0:
            done = True
        else:
            done = False
            
        return reward,done,action,BuyPrice,Budget
    
    