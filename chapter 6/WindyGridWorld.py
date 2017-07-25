import numpy as np
import matplotlib.pyplot as plt

wind = np.zeros(10)
wind[3:6] = 1
wind[6:8] = 2
wind[8] = 1


def act_argmax(state, Q):
    max_val = float('-inf')
    for act in range(4):
        if Q[tuple(state), act] > max_val:
            max_val = Q[tuple(state), act]
            result = act
    return result
    

def make_action(state, action):
    new_state = state + action
    new_state[1] += wind[state[0]]
    if new_state[0] < 0:
        new_state[0] = 0
    if new_state[0] > 9:
        new_state[0] = 9
    if new_state[1] < 0:
        new_state[1] = 0
    if new_state[1] > 6:
        new_state[1] = 6
    return new_state

def one_episode(start, target):
    state = np.copy(start)
    time = 0
    while not np.array_equal(state, target):
        time += 1
        if np.random.random() > epsilon:
            action = act_argmax(state, Q)
        else:
            action = np.random.choice([0, 1, 2, 3])
        new_state = make_action(state, actions[action])
        if np.random.random() > epsilon:
            new_action = act_argmax(new_state, Q)
        else:
            new_action = np.random.choice([0, 1, 2, 3])
        reward = -1
        if np.array_equal(new_state, target):
            reward = 0
        Q[tuple(state), action] += alpha * \
            (reward + Q[tuple(new_state), new_action] -
             Q[tuple(state), action])
        state = np.copy(new_state)
        action = new_action
    return time



'''
actions:
    0 = left = (-1, 0)
    1 = up = (0, 1)
    2 = right = (1, 0)
    3 = down = (0, -1)
'''
actions = {0: np.array([-1, 0]),
           1: np.array([0, 1]),
           2: np.array([1, 0]),
           3: np.array([0, -1]),}
#           4: np.array([-1, -1]),
#           5: np.array([-1, 1]),
#           6: np.array([1, -1]),
#           7: np.array([1, 1])}
actions_words = {0: 'left',
                 1: 'up',
                 2: 'right',
                 3: 'down',}
#                 4: 'left_down',
#                 5: 'left_up',
#                 6: 'right_down',
#                 7: 'right_up'}

Q = {}
for i in range(10):
    for j in range(7):
        for act in range(4):
            Q[((i, j), act)] = 0.0

start = np.array([0, 3])
target = np.array([7, 3])

alpha = 0.5
epsilon = 0.1
episodes_times = []
while sum(episodes_times) < 8000:
    episodes_times.append(one_episode(start, target))

episodes = []
time = np.array(range(sum(episodes_times)))
ep = 0
for ep_time in episodes_times:
    for t in range(ep_time):
        episodes.append(ep)
    ep += 1
    
plt.plot(time, episodes)
plt.xlabel('Time steps')
plt.ylabel('Episodes')

state = np.copy(start)
while not np.array_equal(state, target):
    action = act_argmax(state, Q)
    print("s: %s, a: %s" % (state, actions_words[action]))
    state = make_action(state, actions[action])