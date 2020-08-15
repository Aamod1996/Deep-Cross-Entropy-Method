# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 16:05:57 2020

@author: Aamod Save
"""


'''
############################################
#          INITIALIZING LIBRARIES          #
############################################
'''

import gym
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier

#Initialize hyerparameters
n_sessions = 100
percentile = 70
n_episodes = 50
n_test = 2
t_max = 10**3


'''
################################################################
#          DEFINING FUNCTIONS FOR CROSSENTROPY METHOD          #
################################################################
'''

#Function to generate observations for n_sessions
def create_session(agent, t_max=1000):
    
    states,actions = [],[]
    total_reward = 0
    
    s = env.reset()
    
    for t in range(t_max):
        
        #Create probabilities for actions
        probs = agent.predict_proba([s])[0] 
        
        #Choose actions according to their probabilities
        a = np.random.choice(len(probs), p=probs)
        
        #Perform the action
        new_s, r, done, info = env.step(a)
        
        states.append(s)
        actions.append(a)
        total_reward += r
        
        s = new_s
        if done: 
            break
        
    return states, actions, total_reward

#Function to choose top percentile observations
def select_elites(states_batch,actions_batch,rewards_batch,percentile=50):
    
    #Choose the threshold for the reward based on percentile
    reward_threshold = np.percentile(rewards_batch, percentile)
    
    #Choose the states and actions which are above the threshold
    elite_states  = [s for i in range(len(states_batch)) if rewards_batch[i] >= reward_threshold for s in states_batch[i]]
    elite_actions = [a for i in range(len(actions_batch)) if rewards_batch[i] >= reward_threshold for a in actions_batch[i]]
    
    return elite_states, elite_actions

#Function to plot the rewards gained per episode
def show_progress(batch_rewards, log, percentile, reward_range=[-990,+10]):
    
    mean_reward, threshold = np.mean(batch_rewards), np.percentile(batch_rewards, percentile)

    print("mean reward = %.3f, threshold=%.3f"%(mean_reward, threshold))
    plt.subplot(1,2,1)
    plt.plot(list(zip(*log))[0], label='Mean rewards')
    plt.plot(list(zip(*log))[1], label='Reward thresholds')
    plt.legend()
    plt.grid()
    
    plt.subplot(1,2,2)
    plt.hist(batch_rewards, range=reward_range);
    plt.vlines([np.percentile(batch_rewards, percentile)], [0], [100], label="percentile", color='red')
    plt.legend()
    plt.grid()

    plt.show()
    
    
'''
###############################
#          MAIN LOOP          #
###############################
'''
    
if __name__ == '__main__':

    #Create the gym environment
    env = gym.make("CartPole-v0").env  
    env.reset()
    n_actions = env.action_space.n
    
    plt.imshow(env.render("rgb_array"))
    
    #Create the agent
    agent = MLPClassifier(hidden_layer_sizes=(20,20),
                          activation='tanh',
                          warm_start=True, 
                          max_iter=1 
                          )
    
    #Initialize the agent
    agent.fit([env.reset()]*n_actions, list(range(n_actions)))    
    
    
    '''
    ####################################
    #          TRAINING PHASE          #
    ####################################
    '''
    
    #Begin training 
    log = []
    
    for i in range(n_episodes):
        #Generate new n_sessions number of sessions
        sessions = [create_session(agent) for _ in range(n_sessions)]
    
        #Get the observations
        batch_states,batch_actions,batch_rewards = map(np.array, zip(*sessions))
    
        #Get the best states and actions
        elite_states, elite_actions = select_elites(batch_states, batch_actions, batch_rewards, percentile)
        
        #Train the agent on these elite observations
        agent.fit(elite_states, elite_actions)
        
        #Keep a tab of rewards
        mean_reward, threshold = np.mean(batch_rewards), np.percentile(batch_rewards, percentile)
        log.append([mean_reward, threshold])

    #Plot the rewards gained per episode and check the progress
    show_progress(batch_rewards, log, percentile, reward_range=[-1000, 1000])
    
    
    '''
    ###################################
    #          TESTING PHASE          #
    ###################################
    '''
    
    #Watch your agent play
    for i in range(n_test):
        
        total_reward = 0
        
        s = env.reset()
        
        for t in range(t_max):
            
            #Display the environment
            plt.imshow(env.render("rgb_array"))
            
            #Get probabilities for actions
            probs = agent.predict_proba([s])[0]
            
            #Choose an action
            a = np.random.choice(n_actions, p=probs)
        
            #Perform the action
            new_s, r, _, _ = env.step(a)
            
            s = new_s

            total_reward += r
        
        print("Total reward: {}".format(total_reward))
    
    env.close()
    
    
'''
#########################
#          END          #
#########################
'''