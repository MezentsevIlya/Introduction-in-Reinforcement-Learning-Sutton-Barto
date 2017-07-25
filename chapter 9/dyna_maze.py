import numpy as np
import matplotlib.pyplot as plt


'''
actions:
    0 = down
    1 = up
    2 = left
    3 = right
'''
actions = {0: np.array([1, 0]),
           1: np.array([-1, 0]),
           2: np.array([0, -1]),
           3: np.array([0, 1])}


def get_action(Q, state, epsilon):
    if np.random.random() < epsilon:
        return np.random.choice(list(actions.keys()))
    elif np.array_equal(Q[state[0], state[1], :], np.zeros(4)):
        return np.random.choice(list(actions.keys()))
    else:
        return np.argmax(Q[state[0], state[1], :])


def take_action(Q, state, action):
    new_state = state + actions[action]

    new_state[0] = max(min(new_state[0], maze_width - 1), 0)
    new_state[1] = max(min(new_state[1], maze_height - 1), 0)
    reward = 0
    if maze[tuple(new_state)] == 1:
        return state, reward
    elif maze[tuple(new_state)] == 2:
        reward = 1
        return target, reward
    else:
        return new_state, reward


def planning(Q, Model, N, observed_states, taken_action):
    for _ in range(N):
        s_num = np.random.choice(range(len(observed_states)))
        s = observed_states[s_num]
        if taken_action.get(s):
            a = np.random.choice(taken_action[s])
            new_s, r = Model[s[0], s[1], a]
            Q[s[0], s[1], a] += \
                alpha * \
                (
                 r + gamma * np.max(Q[new_s[0], new_s[1], :]) -
                 Q[s[0], s[1], a]
                )
    return Q


def dyna_Q(N, start, target, alpha, gamma, epsilon, episodes, runs):
    steps = np.zeros(episodes)
    for run in range(runs):
        print('Run: %s, N = %s' % (run, N))
        Q = np.zeros((maze_width, maze_height, 4))
        Model = {}
        observed_states = []
        taken_action = {}
        for ep in range(episodes):
            state = np.copy(start)
            while not np.array_equal(state, target):
                steps[ep] += 1
                action = get_action(Q, state, epsilon)
                new_state, reward = take_action(Q, state, action)

                if tuple(state) not in observed_states:
                    observed_states.append(tuple(state))
                    taken_action[tuple(state)] = [action]
                elif action not in taken_action[tuple(state)]:
                    taken_action[tuple(state)].append(action)

                Q[state[0], state[1], action] += \
                    alpha * \
                    (
                     reward +
                     gamma * np.max(Q[new_state[0], new_state[1], :]) -
                     Q[state[0], state[1], action]
                    )

                Model[state[0], state[1], action] = (new_state, reward)

                state = np.copy(new_state)

                planning(Q, Model, N, observed_states, taken_action)

    return steps / runs


def print_optimal_strategy(Q):
    for i in range(maze_width):
        row = []
        for j in range(maze_height):
            act = np.argmax(Q[i, j, :])
            if np.array_equal(Q[i, j, :], np.zeros(4)):
                row.append('0')
            elif act == 0:
                row.append('D')
            elif act == 1:
                row.append('U')
            elif act == 2:
                row.append('L')
            elif act == 3:
                row.append('R')

        print(row)

maze_width = 6
maze_height = 9
maze = np.zeros((maze_width, maze_height))
maze[1, 2] = 1
maze[2, 2] = 1
maze[3, 2] = 1

maze[4, 5] = 1

maze[0, 7] = 1
maze[1, 7] = 1
maze[2, 7] = 1

maze[0, 8] = 2

start = np.array([2, 0])
target = np.array([0, 8])

observed_states = []
taken_action = {}

N = 5
alpha = 0.1
gamma = 0.95
epsilon = 0.1
episodes = 50
runs = 10

steps0 = dyna_Q(0, start, target,
                alpha, gamma, epsilon,
                episodes, runs)
steps5 = dyna_Q(5, start, target,
                alpha, gamma, epsilon,
                episodes, runs)
steps50 = dyna_Q(50, start, target,
                 alpha, gamma, epsilon,
                 episodes, runs)

plt.plot(np.arange(1, len(steps0)), steps0[1:],
         label='N = ' + str(0))
plt.plot(np.arange(1, len(steps5)), steps5[1:],
         label='N = ' + str(5))
plt.plot(np.arange(1, len(steps50)), steps50[1:],
         label='N = ' + str(50))
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.xlabel('Episodes')
plt.ylabel('Step per episode')
