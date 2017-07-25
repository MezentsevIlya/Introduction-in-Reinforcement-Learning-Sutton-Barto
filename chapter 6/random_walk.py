import numpy as np
import matplotlib.pyplot as plt

states_dict = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5}


def get_reward(state):
    if state < 6:
        return 0
    else:
        return 1


def TD_0(v, alpha, gamma=1, s = 'c'):
    s = states_dict[s]
    while True:
        action = np.random.choice([-1, 1])
        new_s = s + action
        reward = 0
        # get_reward(new_s)
        v[s] += alpha * (reward + gamma * v[new_s] - v[s])
        if new_s in [0, 6]:
            break
        else:
            s = new_s
    return v


def MC(v, alpha, gamma=1, s='c'):
    s = states_dict[s]
    states_seen = [s]
    while True:
        action = np.random.choice([-1, 1])
        s = s + action
        states_seen.append(s)
        if s in [0, 6]:
            break
    reward = get_reward(s)
    for state in states_seen[:-1]:
        v[state] += alpha * (reward - v[state])
    return v


def TD_0_batch(v, alpha, gamma=1, s='c'):
    s = states_dict[s]
    states_seen = [s]
    rewards = [0]
    while True:
        action = np.random.choice([-1, 1])
        s = s + action
        states_seen.append(s)
        rewards.append(0)
        if s in [0, 6]:
            break
    return states_seen, rewards


def MC_batch(v, alpha, gamma=1, s='c'):
    s = states_dict[s]
    states_seen = [s]
    while True:
        action = np.random.choice([-1, 1])
        s = s + action
        states_seen.append(s)
        if s in [0, 6]:
            break
    reward = get_reward(s)
    return states_seen, [reward] * len(states_seen)


def RMSError(fig_num, alphas, v_true, states, runs_num, episodes_num, method):
    v = np.zeros(7)
    v[1:6] = 0.5
    v[6] = 1
    plt.figure(fig_num)
    rmse = np.zeros(100)
    for alpha in alphas:
        for run in range(runs_num):
            v_temp = np.copy(v)
            for episode in range(episodes_num):
                v_temp = method(v_temp, alpha)
                rmse[episode] += np.sqrt(np.sum(np.power(v_true[1:-1] -
                                                v_temp[1:-1], 2)) / 5)
        rmse = np.array(rmse) / runs_num
        plt.plot(np.array(range(episodes_num)), rmse,
                 label='alpha = ' + str(alpha))
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.xlabel('RMSE')
        plt.ylabel('Walks/Episodes')


def RMSError_batch(fig_num, alphas, v_true, states, runs_num, episodes_num,
                   method):
    v = np.zeros(7)
    v[1:6] = 0.5
    v[6] = 1
    plt.figure(fig_num)
    rmse = np.zeros(100)
    for alpha in alphas:
        print(alpha)
        print(method)
        for run in range(runs_num):
            print('Run:', run)
            v_temp = np.copy(v)
            trajectories = []
            rewards = []
            for episode in range(episodes_num):
                if episode % 20 == 0:
                    print('Episode:', episode)
                trajectory, reward = method(v_temp, alpha)
                trajectories.append(trajectory)
                rewards.append(reward)
                while True:
                    updates = np.zeros(7)
                    for trajectory, reward in zip(trajectories, rewards):
                        for i in range(len(trajectory) - 1):
                            if method == TD_0_batch:
                                updates[trajectory[i]] += reward[i] + \
                                    v_temp[trajectory[i + 1]] - \
                                    v_temp[trajectory[i]]
                            else:
                                updates[trajectory[i]] += \
                                    reward[i] - v_temp[trajectory[i]]
                    updates *= alpha
                    if np.sum(np.abs(updates)) < 1e-3:
                        break
                    v_temp += updates
                rmse[episode] += np.sqrt(np.sum(np.power(v_true[1:-1] -
                                                v_temp[1:-1], 2)) / 5)
        rmse = np.array(rmse) / runs_num
        plt.plot(np.array(range(episodes_num)), rmse,
                 label='alpha = ' + str(alpha))
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.xlabel('Walks/Episodes')
        plt.ylabel('RMSE')
        
states = [1, 2, 3, 4, 5]
v_true = np.array([0, 1/6, 2/6, 3/6, 4/6, 5/6, 1])


#v = np.zeros(7)
#v[1:6] = 0.5
#v[6] = 1
#episodes = np.array([0, 1, 10, 100])
#plt.figure(1)
#plt.plot(states, v_true[1:-1], label='true value')
#for i in range(episodes[-1] + 1):
#    if i in episodes:
#        plt.plot(states, v[1:-1], label=str(i) + ' episodes')
#    v = TD_0(v, 0.1, 1)
#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#
#print(v)
#plt.figure(2)
#plt.plot(states, v_true[1:-1], label='true value')
#v_mc = np.zeros(7)
#v_mc[1:6] = 0.5
#v_mc[6] = 1
#for i in range(episodes[-1] + 1):
#    if i in episodes:
#        plt.plot(states, v_mc[1:-1], label=str(i) + ' episodes')
#    v_mc = MC(v_mc, 0.1)
#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#
#print(v_mc)

#RMSError(3, [0.15, 0.1, 0.05], v_true, states, 100, 100, TD_0)
#RMSError(3, [0.01, 0.02, 0.03, 0.04], v_true, states, 100, 100, MC)

RMSError_batch(3, [0.001], v_true, states, 100, 100, TD_0_batch)
RMSError_batch(3, [0.001], v_true, states, 100, 100, MC_batch)



