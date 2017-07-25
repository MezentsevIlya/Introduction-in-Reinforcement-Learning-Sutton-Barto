import numpy as np
import matplotlib.pyplot as plt


'''
actions:
    0 = down
    1 = up
    2 = left
    3 = right
'''
actions = {1: np.array([1, 0]),
           0: np.array([-1, 0]),
           2: np.array([0, -1]),
           3: np.array([0, 1])}


def init_maze():
    maze = np.zeros((maze_width, maze_height))

    ## obstacles
    maze[3, 0:-1] = 1

    maze[0, 8] = 2
    return maze


def block_maze(maze):
    maze[3, 0] = 0
    maze[3, -1] = 1
    return maze


def get_action(Q, state, epsilon):
    if np.random.random() < epsilon:
        return np.random.choice(list(actions.keys()))
    elif np.array_equal(Q[state[0], state[1], :], np.zeros(4)):
        return np.random.choice(list(actions.keys()))
    else:
        return np.argmax(Q[state[0], state[1], :])


def take_action(maze, Q, state, action):
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


def get_sample(Model):
    s_index = np.random.choice(range(len(Model.keys())))
    s = list(Model.keys())[s_index]

    a_index = np.random.choice(range(len(Model[s].keys())))
    a = list(Model[s])[a_index]

    new_s, r = Model[s[0], s[1]][a]

    return list(s), a, list(new_s), r


def planning(Q, Model, N):
    for _ in range(N):
        s, a, new_s, r = get_sample(Model)

        Q[s[0], s[1], a] += \
            alpha * \
            (r + gamma * np.max(Q[new_s[0], new_s[1], :]) -
             Q[s[0], s[1], a])
    return Q


def get_sample_plus(Model, k, time):
    s_index = np.random.choice(range(len(Model.keys())))
    s = list(Model.keys())[s_index]

    a_index = np.random.choice(range(len(Model[s].keys())))
    a = list(Model[s])[a_index]

    new_s, r, t = Model[s][a]
    r += k * np.sqrt(time - t)

    return list(s), a, list(new_s), r


def planning_plus(Q, Model, N, k, time):
    for _ in range(N):
        s, a, new_s, r = get_sample_plus(Model, k, time)

        Q[s[0], s[1], a] += \
            alpha * \
            (r + gamma * np.max(Q[new_s[0], new_s[1], :]) -
             Q[s[0], s[1], a])

    return Q


def dyna_Q(N, start, target, alpha, gamma, epsilon, max_time, runs):
    cumulative_rewards = np.zeros(max_time)
    for run in range(runs):
        print('Run: %s, N = %s' % (run, N))
        Q = np.zeros((maze_width, maze_height, 4))
        Model = dict()
        maze = init_maze()
        state = np.copy(start)
        total_reward = 0
        t = 0
        while t < max_time:
            t += 1
            if t == 1000:
                maze = block_maze(maze)
            action = get_action(Q, state, epsilon)
            new_state, reward = take_action(maze, Q, state, action)

            Q[state[0], state[1], action] += \
                alpha * \
                (reward +
                 gamma * np.max(Q[new_state[0], new_state[1], :]) -
                 Q[state[0], state[1], action])

            total_reward += reward
            cumulative_rewards[t - 1] += total_reward

            if tuple(state) not in Model.keys():
                Model[tuple(state)] = dict()
            Model[tuple(state)][action] = [list(new_state), reward]

            if np.array_equal(target, new_state):
                state = np.copy(start)
            else:
                state = np.copy(new_state)

            planning(Q, Model, N)

#        print_optimal_strategy(Q)
    return cumulative_rewards / runs


def dyna_Q_plus(N, k, start, target, alpha, gamma, epsilon,
                max_time, runs):
    cumulative_rewards = np.zeros(max_time)
    for run in range(runs):
        print('Run: %s, N = %s' % (run, N))
        Q = np.zeros((maze_width, maze_height, 4))
        Model = dict()
        maze = init_maze()
        state = np.copy(start)
        total_reward = 0
        t = 0
        while t < max_time:
            t += 1
            if t == 1000:
                maze = block_maze(maze)
            action = get_action(Q, state, epsilon)
            new_state, reward = take_action(maze, Q, state, action)

            total_reward += reward
            cumulative_rewards[t - 1] += total_reward

            Q[state[0], state[1], action] += \
                alpha * \
                (reward +
                 gamma * np.max(Q[new_state[0], new_state[1], :]) -
                 Q[state[0], state[1], action])

            if tuple(state) not in Model.keys():
                Model[tuple(state)] = dict()
                for action_ in actions.keys():
                    if action_ != action:
                        Model[tuple(state)][action_] = [list(state), 0, 1]
            Model[tuple(state)][action] = [list(new_state), reward, t]

            planning_plus(Q, Model, N, k, t)
            if np.array_equal(target, new_state):
                state = np.copy(start)
            else:
                state = np.copy(new_state)

#        print_optimal_strategy(Q)
    return cumulative_rewards / runs


def print_optimal_strategy(Q):
    for i in range(maze_width):
        row = []
        for j in range(maze_height):
            act = np.argmax(Q[i, j, :])
            if np.array_equal(Q[i, j, :], np.zeros(4)):
                row.append('0')
            elif act == 0:
                row.append('U')
            elif act == 1:
                row.append('D')
            elif act == 2:
                row.append('L')
            elif act == 3:
                row.append('R')
        print(row)

maze_width = 6
maze_height = 9
#maze = init_maze()

start = np.array([maze_width - 1, 3])
target = np.array([0, 8])

N = 5
alpha = 0.7
gamma = 0.95
epsilon = 0.1
max_time = 3000
runs = 3

#np.random.seed(1)
#cumulative_rewards50 = dyna_Q(N, start, target,
#                              alpha, gamma, epsilon,
#                              max_time, runs)

cumulative_rewards50_plus = dyna_Q_plus(N, 1e-4, start, target,
                                        alpha, gamma, epsilon,
                                        max_time, runs)

plt.plot(np.arange(len(cumulative_rewards50)), cumulative_rewards50,
         label='N = ' + str(5))
plt.plot(np.arange(len(cumulative_rewards50_plus)), cumulative_rewards50_plus,
         label='N+ = ' + str(5))
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.xlabel('Time steps')
plt.ylabel('Step per episode')