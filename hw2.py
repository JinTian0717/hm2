# %%
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

# binomial tree
N = 6
rfr = 0.0
volatility = 0.2
lamb = 0.5
u = np.e**(volatility*np.sqrt(1/N))
d = 1/u
p = (np.e**(rfr*np.sqrt(1/N)) - d)/(u - d)

S0 = 1
# probability for exploration
EPSILON = 0.1

# step size
ALPHA = 0.2

# gamma for Q-Learning and Expected Sarsa
GAMMA = 1

# all possible actions
ACTION_NO = 2
ACTION_HALF = 1
ACTION_EMPTY = 0

ACTIONS = [ACTION_EMPTY, ACTION_HALF, ACTION_NO]

# %%
# initial state action pair values
states = np.zeros([N,N])
for i in range(N):
    for j in range(i+1):
        states[j,i] = S0*(u**(i-j))*(d**j)

START = [0,0]

# %%
def step(state, action, holdings = 1):
    i, j = state

    if action == ACTION_NO:
        holding = holdings
    elif action == ACTION_HALF:
        holding = max(holdings - 0.5,0)
    elif action == ACTION_EMPTY:
        holding = max(holdings - 1,0)
    else:
        assert False
    
    if np.random.binomial(1, p) == 1:
        next_state = [i, j+1]
    else:        
        next_state = [i+1, j+1]
    
    
    Step_mu = p*np.log(u) + (1-p)*np.log(d)
    Step_sigma = p*(np.log(u)-Step_mu)**2 + (1-p)*(np.log(d)-Step_mu)**2
    reward = holding * Step_mu - lamb * (holding) *(Step_sigma**0.5) + (1-holding)*(np.exp(rfr/N)-1)

    return next_state, reward

# choose an action based on epsilon greedy algorithm
def choose_action(state, q_value):
    
    if np.random.binomial(1, EPSILON) == 1:
        return np.random.choice(ACTIONS)
    else:
        values_ = q_value[state[0], state[1]]
        return np.random.choice([action_ for action_, value_ in enumerate(values_) if value_ == np.max(values_)])

def q_learning(q_value, step_size=ALPHA):
    state = START
    rewards = 0.0
    reward_process = []
    while state[1] <= N-1:
        if state[1] == N-1:
            break
        else:
            action = choose_action(state, q_value)
            next_state, reward = step(state, action)
            rewards += reward
            reward_process.append(reward)
            # rewards =  - 100 * np.std(reward_process)

            # Q-Learning update
            q_value[state[0], state[1], action] += step_size * (
                        reward + GAMMA * np.max(q_value[next_state[0], next_state[1], :]) -
                        q_value[state[0], state[1], action])
            state = next_state

    return q_value


# %%
# print optimal policy
def print_optimal_policy(q_value):
    optimal_policy = []
    for i in range(0, N):
        optimal_policy.append([])
        for j in range(i, N):
            if j == N-1:
               optimal_policy[-1].append('END')
               continue
            bestAction = np.argmax(q_value[i, j, :])
            if bestAction == ACTION_NO:
                optimal_policy[-1].append(0)
            elif (bestAction == ACTION_HALF):
                optimal_policy[-1].append(-0.5)
            elif (bestAction == ACTION_EMPTY):
                optimal_policy[-1].append(-1)
    for row in optimal_policy:
        print(row)


# %% This the the computation of optimal policy
rfrs = [0.0, 0.05, 0.1]
lambs = [0.2,0.4,0.6]
for rfr in rfrs:
    for lamb in lambs:
        volatility = 0.2
        u = np.e**(volatility*np.sqrt(1/N))
        d = 1/u
        p = (np.e**(rfr*np.sqrt(1/N)) - d)/(u - d)

        q_q_learning = np.zeros((N, N, 3))
        for i in range(50000):
            q_q_learning = q_learning(q_q_learning)
        
        print('\nR=',rfr, 'lambda = ',lamb)
        print_optimal_policy(q_q_learning)
# %%
