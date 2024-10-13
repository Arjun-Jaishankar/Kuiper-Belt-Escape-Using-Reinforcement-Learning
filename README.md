# My-IVLabs-Project-Reinforcement_Learning
# Overview
# Task-1 : Frozen Lake Environment Using Dynamic Programming (DP)
## Documentation:-
[Gymnasium Documentation](https://gymnasium.farama.org/environments/toy_text/frozen_lake/#frozen-lake)
## Frozen Lake Environment Overview
### Task Description (Aim):-
The aim of Task-1 is to develop an agent that can successfully navigate a slippery, frozen lake grid environment to reach a goal while avoiding holes. The agent must learn an optimal policy using dynamic programming techniques like Policy Iteration or Value Iteration to maximize its cumulative reward, despite the uncertain, stochastic transitions caused by the slippery nature of the environment.
### Requirements (Libraries):-
`gym`  
`pygame`  
`numpy`  
`matplotlib`
### Observation Space:-
The Observation Space in the Frozen Lake environment represents all the possible states the agent can observe. Each state corresponds to a specific position on the grid, which in this case is an 8x8 or 4x4 grid that mimics a frozen lake with different types of tiles:

Safe tiles: Where the agent can safely move without falling into a hole.
Hole tiles: These tiles will make the agent "fall," ending the episode.
Goal tile: The target tile the agent must reach to win.
In a 4x4 grid, for example, the observation space consists of 16 discrete states (0 to 15), each representing a unique position on the grid.
Observation Space:

A discrete space representing positions on the grid (e.g., 4x4 or 8x8).
The state corresponds to the agent’s current location on the frozen lake.
### Action Space:-
The Action Space defines all possible actions the agent can take from any given state. For the Frozen Lake environment, the actions correspond to moving in one of four directions:

0: Move Left
1: Move Down
2: Move Right
3: Move Up
Thus, the action space is discrete and consists of 4 possible actions. Each action is an attempt to move the agent in the corresponding direction on the grid. However, due to the slippery nature of the lake, the agent may slide to a different tile than the one intended.
Action Space:

4 discrete actions: move left, move down, move right, and move up.
Each action may result in slipping due to the slippery nature of the lake.
### Transition Matrix:-
The Transition Matrix defines the probabilities of moving from one state to another given a particular action. In the Frozen Lake environment, the grid is slippery, meaning that when the agent tries to move in one direction, it might slip and end up in an unintended direction.

The transition probabilities are not deterministic but stochastic:

The agent might succeed in moving in the desired direction (with some probability).
It might also slide to an adjacent state (with some probability).
For example, if the agent attempts to move right, it may:

Move right with a probability of 0.8.
Slide up, down, or stay in the same position with some lower probability (e.g., 0.1 each).
The transition matrix defines these probabilities for each action from every possible state.
Transition Matrix:

Describes the stochastic movement of the agent. When an action is taken, there is a probability the agent will slide in unintended directions.
This matrix defines the state transition probabilities.
### Rewards:-
The Reward Function assigns numerical values based on the agent’s actions and the resulting states:

Reaching the goal state results in a reward of +1.
Falling into a hole results in a reward of 0 (or a penalty if defined).
Stepping onto a regular frozen tile also typically yields a reward of 0.
The goal of the agent is to maximize its cumulative reward by reaching the goal while avoiding the holes. Since rewards are sparse (only given when the agent reaches the goal), the agent must learn to navigate the grid while dealing with the uncertainty caused by slipping.
Reward Function:

Sparse rewards. Reaching the goal provides a reward of +1.
Falling into a hole or stepping on safe tiles yields a reward of 0.
### About The Algorithm:-
In the context of the Frozen Lake task, Dynamic Programming (DP) techniques are used to compute the optimal policy for the agent. Specifically, the following algorithms are commonly applied:

a. Policy Iteration
Policy Evaluation: For a given policy, calculate the value function that estimates how good it is to be in each state under that policy. This is done iteratively using the Bellman equation, which updates the value of a state based on the expected rewards from future states.

Policy Improvement: Based on the evaluated value function, improve the policy by choosing actions that maximize the expected future rewards for each state. The goal is to find a greedy policy that leads to optimal behavior.

Convergence: This process of alternating between policy evaluation and policy improvement continues until the policy no longer changes, i.e., when it converges to the optimal policy.

b. Value Iteration
Instead of performing separate policy evaluation and improvement steps, Value Iteration combines them into a single process:

At each step, the value of a state is updated by taking the maximum expected reward over all possible actions.
The policy is implicitly improved as the value function is updated.
This process continues until the value function converges, leading to an optimal policy where the agent maximizes the expected future rewards.
Dynamic Programming Algorithm:

Policy Iteration: Alternating between policy evaluation and policy improvement to converge on the optimal policy.
Value Iteration: Updating the value function by taking the maximum expected reward for each state and action pair until convergence.
## Results:-
![output](https://github.com/user-attachments/assets/7177c343-81ba-4607-962c-a6d4cad78724)
![output1](https://github.com/user-attachments/assets/34f9a444-38f2-4c8e-ab6e-4f11a6b6f0db)
![image](https://github.com/user-attachments/assets/232000c2-303f-478f-a72b-3fab1e9e09af)
![image](https://github.com/user-attachments/assets/c264ff94-88f6-463b-9b58-eaed4f382b13)
![image](https://github.com/user-attachments/assets/c5871134-74d3-4559-b723-255a5b28d7ba)
![image](https://github.com/user-attachments/assets/6c3ebcd4-7094-40bc-bdbf-6779ac3eedbd)
![image](https://github.com/user-attachments/assets/9c8ca600-ce58-442a-b2a8-2e1ee579aa69)

# Task-2 : Minigrid Environment (Implementing Model-Free Control Algorithms)
## Documentation:-
[Minigrid Empty Space Documentation](https://minigrid.farama.org/environments/minigrid/EmptyEnv/)
## Minigrid Environment Overview
### Task Description (Aim):-
The aim of Task-2 is to train an agent to effectively navigate and interact within a grid-based environment using the MiniGrid framework. The agent's objective is to explore the grid, find a specified target object, and learn to reach that object while avoiding obstacles and utilizing available resources. This task focuses on developing robust navigation and decision-making skills, allowing the agent to learn optimal policies through trial and error, maximizing its cumulative rewards in the process.

### Requirements:-
Try the following in IDE before implementing the algorithms:-
```import gym
import numpy as np 
import random

env = gym.make('MiniGrid-Empty-6x6-v0',render_mode='human')
env.reset()

for i in range(50):
    env.render()
    action = random.randint(0, 3)
    print("action:",action)
    obs=env.step(action)
    # print(obs)
    print(env.agent_pos,'\n')

env.close()
```
### Observation Space:-
The Observation Space in the MiniGrid environment encompasses a variety of information that the agent can perceive while navigating the grid. This includes:

Agent's Position: The current coordinates of the agent within the grid, indicating its location.
Grid Layout: The entire view of the grid, which may contain walls, doors, objects, and other agents. This can be represented as a 2D array or a visual representation.
Object States: Information about nearby objects, including their types (e.g., walls, keys, goals) and states (e.g., whether they are accessible or not).
Agent's Orientation: The direction the agent is facing (e.g., up, down, left, right), which influences its movement and interaction with objects.
In the MiniGrid environment, the observation space can be either discrete or continuous, depending on the specific configuration of the task, and it can be represented as a pixel grid or as a structured set of features.
The observation space includes the agent's position, grid layout, object states, and the agent's orientation. It can be represented as a 2D array or a visual representation of the grid, providing the agent with essential information for navigation and interaction.
### Action Space:-
The Action Space in the MiniGrid environment defines the set of actions available to the agent at any given state. Typical actions include:

Move Forward: The agent moves one step in the direction it is facing.
Turn Left: The agent rotates 90 degrees to the left without moving.
Turn Right: The agent rotates 90 degrees to the right without moving.
Pick Up Object: The agent picks up an object if it is in the same location.
Drop Object: The agent drops an object in its current location if it is holding one.
Open Door: The agent opens a door if it is adjacent to it.
The action space is typically discrete, and the agent must choose its actions wisely to navigate the grid efficiently, overcome obstacles, and reach its target.
The action space consists of discrete actions: moving forward, turning left/right, picking up objects, dropping objects, and opening doors. The agent must strategically choose actions to navigate the grid and reach the target.
### Reward Function:-
The Reward Function in the MiniGrid environment provides feedback to the agent based on its actions and outcomes. The rewards are structured to encourage the desired behavior and can include:

Positive Reward for Reaching the Target: The agent receives a significant positive reward (e.g., +1) for successfully reaching the specified target object or goal.
Negative Reward for Hitting Obstacles: The agent may receive a negative reward (e.g., -1) for colliding with walls or obstacles, discouraging such actions.
Rewards for Picking Up or Dropping Objects: The agent may receive small positive rewards for successfully picking up or dropping objects, promoting exploration and interaction with the environment.
Time Penalty: To encourage efficiency, the agent might receive a small negative reward for each time step taken, incentivizing it to reach the target quickly.
The design of the reward function is crucial for guiding the agent’s learning process, shaping its policy, and ultimately achieving the task's goal.
The reward function provides feedback to the agent, including positive rewards for reaching the target, negative rewards for hitting obstacles, and rewards for picking up or dropping objects. A time penalty may also apply, encouraging efficient navigation to maximize cumulative rewards.
### About The Algorithms:-
1. Monte Carlo Method
The Monte Carlo method is a reinforcement learning algorithm that relies on sampling to estimate the value of states or state-action pairs. It uses episode-based learning, meaning it evaluates the value of an action by observing the total reward received after completing an episode, which consists of a sequence of states and actions that ends in a terminal state. After each episode, the algorithm updates the value function by calculating the average of the returns for each state-action pair observed during that episode. While Monte Carlo methods are simple to implement and understand and work well for episodic tasks, they require many episodes to converge to optimal value estimates and cannot be effectively applied in environments where episodes can be infinitely long or where states are non-terminating.

2. SARSA (State-Action-Reward-State-Action)
SARSA is an on-policy temporal-difference (TD) learning algorithm that updates the value of actions based on the current policy. It evaluates and improves the policy being used to generate data, allowing the algorithm to learn from the actual actions taken, including exploration steps. The algorithm observes the reward and the next state after taking an action, and then updates the Q-value based on the next action chosen according to the current policy. This means that SARSA incorporates exploration directly into the learning process, which can lead to more gradual learning. However, this can also result in slower convergence compared to off-policy methods like Q-learning, as the policy may converge to suboptimal solutions due to the influence of exploration.

3. SARSA(λ)
SARSA(λ) extends the SARSA algorithm by integrating eligibility traces, which allow for more efficient updates. The use of eligibility traces combines the benefits of both Monte Carlo and TD methods, maintaining a trace for each state-action pair that decays over time. This enables SARSA(λ) to update not just the last action taken, but all actions in the eligibility trace, allowing the algorithm to learn from multiple past experiences. While this can lead to faster convergence than SARSA alone, it introduces additional complexity due to the management of eligibility traces and requires careful tuning of the trace decay parameter to balance the learning process effectively.

4. Q-Learning
Q-learning is an off-policy reinforcement learning algorithm that aims to learn the optimal action-value function independently of the policy being followed. This means that Q-learning evaluates the value of the optimal policy, regardless of the actions taken by the agent, and it updates the Q-values based on the maximum estimated future rewards. This approach allows Q-learning to converge to the optimal policy even when the agent is exploring the environment in a suboptimal manner. While this off-policy nature offers great flexibility, it may require more exploration to balance out the learning, especially in non-stationary environments. Nevertheless, Q-learning is typically simpler to implement and tune compared to on-policy methods, making it a more efficient choice for complex environments.
## Results:-
### Monte Carlo:-
![MC Learning No  of Steps Vs Episodes](https://github.com/user-attachments/assets/f9104fad-da89-4fd5-b9be-9985aa419a4b)
![MC Learning Returns Vs Episodes](https://github.com/user-attachments/assets/37910d1d-5e4b-47c8-9006-b866cfa1ebda)
### Q-Learning:-
![Q-learning Steps Vs Episodes](https://github.com/user-attachments/assets/3c005814-db22-4943-9c22-1f67109768f3)
![Q-learning Return Vs Episodes](https://github.com/user-attachments/assets/6f38515b-90a4-406e-929d-94be2568e566)
### Sarsa:-
![Sarsa Steps Vs Episode](https://github.com/user-attachments/assets/ea179368-17b1-4baf-9626-85291ca1e2ad)
![Sarsa Reward Vs Episode](https://github.com/user-attachments/assets/e7941232-9a81-4ff1-9da1-f2bdb2b9c7be)
### Sarsa(λ):-
![Sarsa Lambda Steps Vs Episode](https://github.com/user-attachments/assets/6b05f54b-c240-4deb-b9c0-0aa7a8b7f574)
![Sarsa Lambda Reward Vs Episode](https://github.com/user-attachments/assets/13902817-941a-474d-87b8-2fa1cb61a3c8)
## Why Prefer Q-Learning for the Kuiper Belt Escape Project
For the Kuiper Belt Escape Project, Q-learning is favored due to its strengths in learning optimal policies and robustness to exploration strategies. The off-policy nature of Q-learning allows it to learn the optimal policy independently of the actions taken by the agent, which is crucial in a dynamic and unpredictable environment like the Kuiper Belt. The algorithm’s capability to handle large state spaces efficiently enables timely decision-making, essential for navigating complex scenarios. Additionally, Q-learning’s simplicity of implementation and adaptability to stochastic environments make it a practical choice for rapid prototyping and experimentation in the project. Overall, Q-learning’s advantages in policy learning, exploration robustness, and ease of use position it as the preferred algorithm for the Kuiper Belt Escape Project.
