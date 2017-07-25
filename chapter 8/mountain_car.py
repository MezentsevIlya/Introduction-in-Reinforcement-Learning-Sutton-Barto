import numpy as np
from TileCoding import IHT, tiles
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

NUM_OF_TILINGS = 8
min_position = -1.2
max_position = 0.5
min_velocity = -0.07
max_velocity = 0.07
actions = [-1, 0, 1]

max_episodes = 104
max_size = 2048
iht = IHT(max_size)


def get_active_tiles(state, action):
    position, velocity = state
    return tiles(iht,
                 NUM_OF_TILINGS,
                 [8*position/(max_position - min_position),
                  8*velocity/(max_velocity - min_velocity)],
                 [action])


def get_value(state, action, weights):
    position, velocity = state
    if position == max_position:
        return 0.0
    active_tiles = tiles(iht,
                         NUM_OF_TILINGS,
                         [8*position/(max_position - min_position),
                          8*velocity/(max_velocity - min_velocity)],
                         [action])
    return np.sum(weights[active_tiles])


def get_action(state, weights, epsilon=0.0):
    if np.random.random() < epsilon:
        return np.random.choice(actions)
    else:
        state_action_values = list(map(lambda a:
                                       get_value(state, a, weights), actions))
        #max_value = max(state_action_values)
        #print(state_action_values)
        #print(state_action_values.index(max_value))
        #print(np.argmax(state_action_values))
#        a = input('a')
        return actions[np.argmax(state_action_values)]


def get_start_state():
    return np.random.uniform(-0.6, -0.4), 0.0

def make_action(state, action):
    position, velocity = state
    new_velocity = velocity + 0.001 * action - 0.0025 * np.cos(3 * position)
    new_velocity = min(max(min_velocity, new_velocity), max_velocity)

    new_position = position + new_velocity
    new_position = min(max(min_position, new_position), max_position)

    reward = -1.0
    if new_position == min_position:
        new_velocity = 0.0
    return [new_position, new_velocity], reward


def sarsa(alpha, epsilon=0.0, gamma=1.0):
    '''
    state = position, velocity
    '''
    weights = np.zeros(max_size)
    steps = []
    for ep in range(max_episodes):
#        print('Episode: %s' % ep)
        state = get_start_state()
        action = get_action(state, weights, epsilon)
        steps.append(0)
        while True:
            steps[-1] += 1
            if steps[-1] % 1000 == 0:
                print('Episode: %s' % ep)
                print('Steps: %s' % steps[-1])
                print('State: %s' % state)
                print('Action: %s' % action)
                print(list(map(lambda a: 
                    get_value(state, a, weights), actions)))
            new_state, reward = make_action(state, action)
            state_value = get_value(state, action, weights)
            active_tiles = get_active_tiles(state, action)
            if new_state[0] == max_position:
                print(new_state)
                delta = reward - state_value
                weights[active_tiles] += alpha * delta
                break
            else:
                new_action = get_action(new_state, weights)
                new_state_value = get_value(new_state, new_action, weights)
                delta = reward + gamma * new_state_value - state_value
                weights[active_tiles] += alpha * delta
                state = new_state
                action = new_action
    return weights, steps


alpha = 0.03
weights, steps = sarsa(alpha)

states = np.arange(min_position, max_position,
                   (max_position - min_position) / 100)
vels = np.arange(min_velocity, max_velocity,
                 (max_velocity - min_velocity) / 100)


def get_max_value(state, weights):
    state_action_values = \
        list(map(lambda a: get_value(state, a, weights), actions))
    return np.max(state_action_values)

m_q = np.zeros((len(states), len(vels)))
for i in range(len(states)):
    for j in range(len(vels)):
        m_q[i, j] = -get_max_value([states[i], vels[j]], weights)

fig = plt.figure()
ax = fig.gca(projection='3d')
X, Y = np.meshgrid(vels, states)
surf = ax.plot_surface(X=X, Y=Y, Z=m_q,
                       cmap=cm.coolwarm, linewidth=0, antialiased=False)
        