import numpy as np
import matplotlib.pyplot as plt

n = 10
h = 0.25
p_124 = (1 - h) / 3
p = 0.06
alpha = 0.01
beta = 0.01
epsilon = 0.1


def get_new_client_p():
    return np.random.choice([1, 2, 4, 8], p=[p_124, p_124, p_124, h])


def get_action(Q, free_servers, client_p, epsilon=0.1):
    if free_servers == 0:
        return 0    
    if np.random.random() < epsilon:
#        action = (action + 1) % 2
        return np.random.choice([0, 1])
    else:
        return np.argmax(Q[free_servers, clients[client_p], :])


def get_free_servers_num(free_servers):
    new_num = free_servers
    for i in range(n - free_servers):
        if np.random.random() < p:
            new_num += 1
    return min(new_num, n)


def make_action(Q, free_servers, client_p, epsilon=0.1):
    action = get_action(Q, free_servers, client_p, epsilon)
    reward = 0
    new_free_servers = free_servers - action
    reward = client_p * action
    new_free_servers = get_free_servers_num(new_free_servers)
    new_client_p = get_new_client_p()
    new_action = get_action(Q, new_free_servers, new_client_p, epsilon)
    return action, new_free_servers, new_client_p, new_action, reward

clients = {1: 0,
           2: 1,
           4: 2,
           8: 3}


V = np.zeros(n + 1)
Q = np.zeros((n + 1, 4, 2))
visits = np.zeros((n + 1, 4, 2), dtype=int)

servers_num = n
rho = 0
client_p = np.random.choice([1, 2, 4, 8], p=[p_124, p_124, p_124, h])
free_servers = servers_num
av_rew = 0
for i in range(1000000):
    if i % 10000 == 0:
        print(i)
    action, new_free_servers, new_client_p, new_action, reward = \
        make_action(Q, free_servers, client_p, epsilon)
#    print(action)
#    print(new_free_servers)
#    print(new_client_p)
#    print(new_action)
#    print(reward)
#    print('\n')
    delta = reward - rho + Q[new_free_servers,
                                 clients[new_client_p], new_action] - \
        Q[free_servers, clients[client_p], action]

    Q[free_servers, clients[client_p], action] += alpha * delta
    visits[free_servers, clients[client_p], action] += int(1)
    if action == np.argmax(Q[free_servers, clients[client_p], :]):
        rho += beta * delta

    free_servers = new_free_servers
    client_p = new_client_p

print(av_rew / i)
for i in [1, 2, 4, 8]:
    row = []
    for j in range(1, 11):
        row.append(np.argmax(Q[j, clients[i], :]))
    print(row)

v = [min(Q[0, 0, :])]
for i in range(1, n+1):
    v.append(max(Q[i, 0, :]))
plt.plot(np.array(range(0, n+1)), v, label='priority=1')

v = [min(Q[0, 1, :])]
for i in range(1, n+1):
    v.append(max(Q[i, 1, :]))
plt.plot(np.array(range(0, n+1)), v, label='priority=2')

v = [min(Q[0, 2, :])]
for i in range(1, n+1):
    v.append(max(Q[i, 2, :]))
plt.plot(np.array(range(0, n+1)), v, label='priority=4')

v = [min(Q[0, 3, :])]
for i in range(1, n+1):
    v.append(max(Q[i, 3, :]))
plt.plot(np.array(range(0, n+1)), v, label='priority=8')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)