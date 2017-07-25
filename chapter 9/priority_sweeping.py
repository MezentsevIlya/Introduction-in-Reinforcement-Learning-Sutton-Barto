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


def planning(Q, Model, N):
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


def dyna_Q(N, start, target, alpha, gamma, epsilon, runs):
    total_sweeping = 0
    for run in range(runs):
        print('Run: %s, N = %s' % (run, N))
        Q = np.zeros((maze_width, maze_height, 4))
        state = np.copy(start)
        Model = {}
        while not check_path(Q):
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

            if np.array_equal(target, new_state):
                state = np.copy(start)
            else:
                state = np.copy(new_state)

            planning(Q, Model, N)
            total_sweeping += N

#        print_optimal_strategy(Q)
    return total_sweeping / runs


def priority_sweeping():
    for run in range(runs):
        print('Run: %s, N = %s' % (run, N))
        Q = np.zeros((maze_width, maze_height, 4))
        Model = {}
        state = np.copy(start)
        observed_states = []
        taken_action = {}
        pre_states = {}
        PQueue = {}
        while not check_path(Q):
            action = get_action(Q, state, epsilon)
            new_state, reward = take_action(Q, state, action)
            
            if not pre_states.get(tuple(new_state)):
                pre_states[new_state] = [tuple(state, action)]
            else:
                pre_states[new_state].append(tuple(state, action))

            if tuple(state) not in observed_states:
                observed_states.append(tuple(state))
                taken_action[tuple(state)] = [action]
            elif action not in taken_action[tuple(state)]:
                taken_action[tuple(state)].append(action)

            Model[state[0], state[1], action] = (new_state, reward)

            p = np.abs(reward +
                       gamma * np.max(Q[new_state[0], new_state[1], :]) -
                       Q[state[0], state[1], action])

            if p > teta:
                PQueue[p] = [state, action]

            n = 0
            while n < N and bool(PQueue):
                
            

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


def check_path(Q):
    max_steps = 14 * factor * 1.2
    state = start
    steps = 0
    while state not in target:
        best_action = np.argmax(Q[state[0], state[1], :])
        state, _ = take_action(Q, state, best_action)
        steps += 1
        if steps > max_steps:
            return False
    return True


def init_maze(factor, maze_width, maze_height):
    maze = np.zeros((maze_width, maze_height))
    obstacles = [[1, 2],
                 [2, 2],
                 [3, 2],
                 [4, 5],
                 [0, 7],
                 [1, 7],
                 [2, 7]]
    for i, j in obstacles:
        state = [i * factor, j * factor]
        for l in range(factor):
            for m in range(factor):
                maze[state[0] + l, state[1] + m] = 1

    state = [0 * factor, 8 * factor]
    maze[state[0], state[1]] = 2
    return maze

factor = 2
maze_width = 6 * factor
maze_height = 9 * factor
maze = init_maze(factor, maze_width, maze_height)

start = np.array([2, 0]) * factor
target = np.array([0, 8]) * factor

N = 5
alpha = 0.1
gamma = 0.95
epsilon = 0.1
episodes = 50
runs = 1

total_sweeps5 = dyna_Q(5, start, target,
                       alpha, gamma, epsilon,
                       runs)
total_sweeps50 = dyna_Q(50, start, target,
                        alpha, gamma, epsilon,
                        runs)

print(total_sweeps5)
print(total_sweeps50)
#plt.plot(np.arange(1, len(steps5)), steps5[1:],
#         label='N = ' + str(5))
#plt.plot(np.arange(1, len(steps50)), steps50[1:],
#         label='N = ' + str(50))
#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#plt.xlabel('Episodes')
#plt.ylabel('Step per episode')
