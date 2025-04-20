import gym
import gym_kuiper_escape
import random as rnd
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

import math
env = gym.make('kuiper-escape-base-v0',mode='None',rock_rate=0.3,player_speed=0.5,rock_size_min=0.08,rock_size_max=0.1)
Q={}
num_episodes=10000
# testcase=500
alpha=0.7
epsilon=1
gamma=0.9
# policy={}
count=[]
rt=[]
test=[]
test_return=[]

def epsilon_greedy_policy(state, Q, epsilon):
 
    if np.random.rand() < epsilon:
        return np.random.randint(0,5)  # Random action
    else:
        return np.argmax(Q[state])      #to get greedy action 
    
def sma(rt,num_episodes) :
  Sma=[]
  B=0
  for i in range (num_episodes) :
   for j in range (i-25, i+25):
     if j>=0 and j<num_episodes:
        B+=rt[j]
   Sma.append(B/50) 
  return Sma    

# def plot_graphs(episodes,iterations,total_reward):
#  plt.plot(range(episodes), iterations)
#  plt.xlabel('Episode')
#  plt.ylabel('Steps')
#  plt.title('Steps per Episode')
#  plt.show()
#  plt.plot(range(episodes),total_reward)
#  plt.xlabel('Episode')
#  plt.ylabel('return')
#  plt.title('return per Episode')
#  plt.show()
def plot_graphs(episodes, iterations, total_reward,Sma):
    plt.figure(figsize=(12, 5),dpi=200)
    
    plt.subplot(1, 2, 1)
    plt.plot(range(episodes), iterations)
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.title('Steps per Episode')
    
    plt.subplot(1, 2, 2)
    plt.plot(range(episodes), total_reward)
    plt.plot(range(num_episodes), Sma,color='red') 
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.title('Return per Episode')
    
    plt.tight_layout()
    
    try:
        # Save the plots
        plt.savefig(f'training_plots_and_sma{episodes}.png')

    except Exception as e:
        print(f"Error with plot: {e}")
    finally:
        plt.close()


   
   


# def reward_manipulation(xp,yp):

        # xp = xp /env.game.screen_size
        # yp = yp / env.game.screen_size
        # dist_from_center = math.sqrt((xp-0.5)**2 + (yp-0.5)**2)
        # if dist_from_center < 0.35:
        #     reward = 1
        # else:
        #     reward = -dist_from_center
        # return reward

max_steps = 3000

for episode in range (num_episodes):
    ini_observation=env.reset()
    # observation=np.round(observation,decimals=1)
    state=tuple(ini_observation)
    slope=(1)/num_episodes
    # epsilon_min=0.1
    # epsilon_max=1
    # k=0.3
    # t0=1200

    done = False
    iterations=0
    if (episode%50==1):
       print(f'Epi:{episode} , Reward:{total_return}')
    total_return=0
    while not done and iterations<max_steps:
        # env.render(mode='human')


        if state not in Q:
           Q[state]=np.zeros(5,dtype=float)

        action=epsilon_greedy_policy(state,Q,epsilon)

        observation,reward,done,info=env.step(action)


        
        if done:
           reward = -5

        iterations+=1

        # reward_man=reward_manipulation(observation)
        # reward_man=reward_manipulation(xp,yp)


        next_state=tuple(observation)

        if next_state not in Q:
           Q[next_state]=np.zeros(5)


        Q[state][action]+=alpha*(reward+gamma*np.max(Q[next_state])-Q[state][action]) 

        total_return+=reward


        # if (done or iterations==env.iteration_max):


        state=next_state
    # print(f'{episode}done in {iterations} iterations with reward {total_return}')

    # if episode>=num_episodes:
    #    test.append(iterations)
    #    test_return.append(total_return)
    count.append(iterations)
    rt.append(total_return)


    # epsilon=max(0.01, epsilon * 0.95)
    # epsilon=epsilon_min+(epsilon_max-epsilon_min)/(1+np.exp(-k*(episode-t0)))

    # epsilon=max(0.1,1-slope*episode)
    epsilon=1-1.4*episode/num_episodes
    # alpha=alpha-episode*slope 


# if len(test) != testcase or len(test_return) != testcase:
#     print(f"Warning: Data length mismatch. test: {len(test)}, test_return: {len(test_return)}, expected: {testcase}")
#     # Adjust testcase to match actual data length
#     testcase = min(len(test), len(test_return))  
# plot_graphs(testcase,test,test_return) 
Sma=sma(rt, num_episodes)
plot_graphs(num_episodes,count,rt,Sma)

