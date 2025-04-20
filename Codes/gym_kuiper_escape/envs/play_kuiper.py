import gym_kuiper_escape
import gym
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import math
import matplotlib.pyplot as plt


num_episodes = 3000
alpha = 0.6        
gamma = 0.7      
epsilon = 1.0      
epsilon_decay = 0.99  
min_epsilon = 0.0000001 

env = gym.make('kuiper-escape-base-v0', mode='None',rock_rate=0.2,player_speed=0.5,rock_size_min=0.08,rock_size_max=0.1,)
Q = {}
epi_num=[]
epi_returns=[]

def calculate_sma(data,window_size):
    return np.convolve(data,np.ones(window_size)/window_size,mode='valid')

def epsilon_greedy_policy(state, epsilon):
    state=tuple(state)
    if np.random.rand() < epsilon:
        return np.random.randint(0, 5) 
    else:
        return np.argmax(Q.get(state, np.zeros(5))) 
    

for episode in range(num_episodes):
    obs = env.reset()
    obs = obs.flatten() 
    m=obs[:8]
    m1=tuple(m)
    done = False
    total_reward = 0
    c=0
    iterations = 0

    
    while not done and c<=1000:
        action = epsilon_greedy_policy(obs, epsilon)
        next_obs,rew, done, _ = env.step(action)
        next_obs = next_obs.flatten()
        h=next_obs[:8]
        h1=tuple(h)
        if done:
            rew+=-10  
        iterations +=1

        if m1 not in Q:
            Q[m1] = np.zeros(5)  

        if h1 not in Q:
            Q[h1] = np.zeros(5)

        Q[m1][action] += alpha * (rew + gamma * np.max(Q[h1]) - Q[m1][action])
        
        obs=next_obs 
        total_reward += rew
        c=c+1

    epsilon = max(min_epsilon, epsilon * epsilon_decay)
    epi_returns.append(total_reward)
    print(f"episode",{episode},"rewards",{total_reward})
env.close()

# Sma=sma(epi_returns, num_episodes)
# plot_graphs(num_episodes,count,epi_returns,Sma)

sma_rewards = calculate_sma(epi_returns,120)

plt.figure(figsize=(10, 6), dpi=200) 
plt.subplot(2,1,1)
plt.plot(epi_returns, label='Reward per Episode', color='blue')
plt.xlabel('Episodes')
plt.ylabel('Total Reward')
plt.title('Reward vs. Episodes')
plt.plot(range(119,num_episodes),sma_rewards,label=f'{120}-episodes_sma',color = 'red')
plt.grid(True)
plt.legend()
plt.savefig('reward_vs_episodes.png')

plt.subplot(2,1,2)
plt.plot(range(episode),iterations)
plt.xlabel('Episodes')
plt.ylabel('Steps')
plt.title('Steps vs Episodes')
plt.savefig('steps_vs_episodes.png')