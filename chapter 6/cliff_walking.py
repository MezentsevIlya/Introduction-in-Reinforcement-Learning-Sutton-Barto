'''
grid: 12 x 4
start = (0, 0)
target = (11, 0)
cliff = [1:11, 0]
'''
import numpy as np
import matplotlib.pyplot as plt


def make_action(state, action, start, target):
    new_state = np.array([0, 0])
    new_state[0] = max(min(state[0] + action[0], 11), 0)
    new_state[1] = max(min(state[1] + action[1], 3), 0)
    reward = -1
    if new_state[0] < 11 and new_state[0] > 0 and new_state[1] == 0:
        new_state = np.copy(start)
        reward = -100
#    if np.array_equal(new_state, target):
#        reward = 0
    return new_state, reward


def one_episode_sarsa(action_state_values, start, target, alpha=0.5,
                      epsilon=0.1):
    Q = np.copy(action_state_values)
    state = np.copy(start)
    total_reward = 0.0
    while not np.array_equal(state, target):
        if np.random.random() > epsilon:
            action = np.argmax(Q[state[0], state[1], :])
        else:
            action = np.random.choice([0, 1, 2, 3])
        new_state, reward = make_action(state, actions[action], start, target)
        total_reward += reward
        if np.random.random() > epsilon:
            new_action = np.argmax(Q[new_state[0], new_state[1], :])
        else:
            new_action = np.random.choice([0, 1, 2, 3])
        Q[state[0], state[1], action] += alpha * \
            (reward + Q[new_state[0], new_state[1], new_action] -
             Q[state[0], state[1], action])
        state = np.copy(new_state)
    return Q, total_reward


def one_episode_Q_learning(action_state_values, start, target, alpha=0.5,
                           epsilon=0.1):
    Q = np.copy(action_state_values)
    state = np.copy(start)
    total_reward = 0
    while not np.array_equal(state, target):
        if np.random.random() > epsilon:
            action = np.argmax(Q[state[0], state[1], :])
        else:
            action = np.random.choice([0, 1, 2, 3])
        new_state, reward = make_action(state, actions[action], start, target)
        total_reward += reward
        Q[state[0], state[1], action] += alpha * \
            (reward + np.max(Q[new_state[0], new_state[1], :]) -
             Q[state[0], state[1], action])
        state = np.copy(new_state)
    return Q, total_reward


def printOptimalPolicy(Q):
    optimalPolicy = []
    for i in range(3, -1, -1):
        optimalPolicy.append([])
        for j in range(0, 12):
            if [j, i] == [0, 11]:
                optimalPolicy[-1].append('G')
                continue
            bestAction = np.argmax(Q[j, i, :])
            if bestAction == 1:
                optimalPolicy[-1].append('U')
            elif bestAction == 3:
                optimalPolicy[-1].append('D')
            elif bestAction == 0:
                optimalPolicy[-1].append('L')
            elif bestAction == 2:
                optimalPolicy[-1].append('R')
    for row in optimalPolicy:
        print(row)

actions = {0: np.array([-1, 0]),
           1: np.array([0, 1]),
           2: np.array([1, 0]),
           3: np.array([0, -1])}
actions_words = {0: 'left',
                 1: 'up',
                 2: 'right',
                 3: 'down'}


Q = np.zeros((12, 4, 4))
for i in range(12):
    for j in range(4):
        for act in range(4):
            Q[i, j, act] = 0.0

start = np.array([0, 0])
target = np.array([11, 0])

alpha = 0.5
epsilon = 0.1
runs_num = 20
episodes_num = 500
rewards_sarsa = np.zeros(episodes_num)
rewards_q_learning = np.zeros(episodes_num)
for run in range(runs_num):
    Q_sarsa = np.copy(Q)
    Q_Q_learning = np.copy(Q)
    print(run)
    for ep in range(episodes_num):
        Q_sarsa, reward = one_episode_sarsa(Q_sarsa, start, target)
        rewards_sarsa[ep] += max(reward, -100)
        Q_Q_learning, reward = \
            one_episode_Q_learning(Q_Q_learning, start, target)
        rewards_q_learning[ep] += max(reward, -100)

printOptimalPolicy(Q_sarsa)
printOptimalPolicy(Q_Q_learning)

rewards_sarsa /= runs_num
rewards_q_learning /= runs_num
averageRange = 10
smoothedRewardsSarsa = np.copy(rewards_sarsa)
smoothedRewardsQLearning = np.copy(rewards_q_learning)
for i in range(averageRange, episodes_num):
    smoothedRewardsSarsa[i] = np.mean(rewards_sarsa[i - averageRange: i + 1])
    smoothedRewardsQLearning[i] = \
        np.mean(rewards_q_learning[i - averageRange: i + 1])

plt.plot(np.array(range(episodes_num)), smoothedRewardsSarsa, label='SARSA')
plt.plot(np.array(range(episodes_num)), smoothedRewardsQLearning,
         label='Q-learning')
plt.xlabel('Time steps')
plt.ylabel('Episodes')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

state = np.copy(start)
while not np.array_equal(state, target):
    action = np.argmax(Q_sarsa[state[0], state[1], :])
    print("s: %s, a: %s" % (state, actions_words[action]))
    state, _ = make_action(state, actions[action], start, target)

state = np.copy(start)
while not np.array_equal(state, target):
    action = np.argmax(Q_Q_learning[state[0], state[1], :])
    print("s: %s, a: %s" % (state, actions_words[action]))
    state, _ = make_action(state, actions[action], start, target)
