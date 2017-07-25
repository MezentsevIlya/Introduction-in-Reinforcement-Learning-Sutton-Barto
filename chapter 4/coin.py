import matplotlib.pyplot as plt
import numpy as np

S = np.array([x for x in range(101)])
print(S)

s = 10

A = np.array([x for x in range(1, min(s, 100 - s) + 1)])
print(A)

reward = {100: 1}
for i in range(100):
    reward[i] = 0

p = 0.4
s = 10

delta = 0
theta = 0.0
gamma = 1
V = np.zeros(101)
V[100] = 0
k = 0
for k in range(32):
    delta = 0
    V_old = V.copy()
    for s in S[1:-1]:
        v = V_old[s]
        A = np.array([x for x in range(1, min(s, 100 - s) + 1)])
        max_v = np.zeros(len(A))
        for a in A:
            new_s = s + a
            max_v[a - 1] = p * (reward.get(new_s) + gamma * V[new_s]) + \
                (1 - p) * (gamma * V[s - a])
        V[s] = max(max_v)
        delta = max(delta, abs(v - V[s]))
    if k in [0, 1, 2, 31]:
        plt.figure(1)
        V_plot = V.copy()
        V_plot[100] = 1
        plt.plot(V_plot, linewidth=1)
#        print(delta)
    print('k = %s' % k)
#    print('V: %s' % V)

    if delta < theta:
        break

plt.figure(3)
V_plot = V.copy()
V_plot[100] = 1
plt.plot(V_plot, linewidth=1)

optimal_str = np.zeros(101)
optimal_v_str = np.zeros(101)

for s in range(1, 100):
    sum = 0
    max_v = 0
    opt_a = 0
    A = np.array([x for x in range(1, min(s, 100 - s) + 1)])
    for a in A:
        new_s = a + s
        new_v = p * (reward.get(new_s) + gamma * V[new_s]) + \
            (1 - p) * (reward.get(s - a) + gamma * V[s - a])
        if new_v > max_v + 0.000000000001:
#            print("s: %s" % s)
#            print("a: %a" % a)
#            print("opt_a: %s" % opt_a)
#            print("max_v: %.20f" % max_v)
#            print("new_v: %.20f" % new_v)
            opt_a = a
            max_v = new_v
#    print("s: %s" % s)
#    print("a: %s" % a)
#    print("max_v: %s" % max_v)
    optimal_str[s] = opt_a
    optimal_v_str[s] = max_v


print(optimal_str)
print(optimal_v_str)

plt.figure(2)
plt.plot(optimal_str, linewidth=1)


























