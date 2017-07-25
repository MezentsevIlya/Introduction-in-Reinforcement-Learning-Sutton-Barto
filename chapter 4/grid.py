import numpy as np


def print_V(V, row_len, col_len):
    for i in range(col_len):
        print(V[i*row_len: (i+1)*row_len])


def get_new_V_s(s, V_old, action_num, pi, P, R, gamma):
    state_num = len(V_old)
    sum = 0
    for a in range(action_num):
        for new_s in range(state_num):
            if P.get((s, a)) == new_s:
                sum += pi[s, a] * (R + gamma * V_old[new_s])
    return sum


def count_V_pi(max_iter, V, action_num, pi, P, R, gamma, theta):
    state_num = len(V)
    V_new = V.copy()
    for k in range(max_iter):
        delta = 0
        V_old = V_new.copy()
        for s in range(1, state_num - 1):
            v = V_old[s]
            V_new[s] = get_new_V_s(s, V_old, action_num, pi, P, R, gamma)
            delta = max(delta, abs(v - V_new[s]))
        print('k = %s' % k)
        print_V(V_new, row_len, col_len)
        if delta < theta:
            return V_new

state_num = 16
action_num = 4
row_len = 4
col_len = 4

V = np.zeros(state_num)
print(V)

actions = {0: 'up', 1: 'down', 2: 'right', 3: 'left'}

pi = np.zeros((state_num, action_num))
for s in range(1, state_num - 1):
    for a in range(action_num):
        pi[s, a] = 1 / action_num

print(pi)

R = -1

P = {}

for i in range(action_num):
    for s in range(1, state_num - 1):
        if i == 0:
            if (s - row_len) < 0:
                P[(s, i)] = s
            else:
                P[(s, i)] = s - row_len
        elif i == 1:
            if (s + row_len) > state_num - 1:
                P[(s, i)] = s
            else:
                P[(s, i)] = s + row_len
        elif i == 2:
            if (s + 1) % row_len == 0:
                P[(s, i)] = s
            else:
                P[(s, i)] = s + 1
        elif i == 3:
            if s % row_len == 0:
                P[(s, i)] = s
            else:
                P[(s, i)] = s - 1

print(P)
V_old = np.zeros(state_num)
gamma = 1
theta = 0.0001

V = count_V_pi(10000, V, action_num, pi, P, R, gamma, theta)
print_V(V, 4, 4)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    