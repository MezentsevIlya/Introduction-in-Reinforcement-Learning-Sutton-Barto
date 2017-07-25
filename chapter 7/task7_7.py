import numpy as np

actions = {0: 0,
           1: 0,
           2: 0,
           3: 1}

def get_eps_greedy_action(Q, s, epsilon=0.1):
    if np.random.random() < epsilon:
        return np.random.choice([0, 1, 2, 3])
    else:
        return np.argmax(Q[s, :])

def sarsa_lambda(Q, alpha, lambda_, epsilon=0.1, gamma=1):
    e = np.zeros([states_num + 1, len(actions)])
    state = 0
    while state < target:
#        print(state)
        action = get_eps_greedy_action(Q, state, epsilon)
        new_state = state + actions[action]
        reward = rewards[new_state]
        new_action = get_eps_greedy_action(Q, new_state, epsilon)
        delta = reward + gamma * Q[new_state, new_action] - Q[state, action]
        e[state, action] += 1
        
        Q += alpha * delta * e
        e *= gamma * lambda_
        e[state] = 1
        state = new_state
    return Q
    
    
states_num = 20
rewards = np.zeros(states_num + 1)
rewards[-1] = 1
start = 0
target = states_num

episodes_num = 100
Q = np.zeros([states_num + 1, len(actions)])
alphas = np.arange(0, 1.1, 0.1)
lambda_ = 0.9
epsilon = 0.1
for alpha in alphas:
    print('alpha = %s' % alpha)
    for ep in range(episodes_num):
        Q = sarsa_lambda(Q,
                         alpha=alpha,
                         lambda_=lambda_,
                         epsilon=epsilon)
    for state in range(states_num):
        print('State: %s, Action: %s', (state, np.argmax(Q[state, :])))
    a = input('a')

for state in range(states_num):
    print('State: %s, Action: %s', (state, np.argmax(Q[state, :])))
