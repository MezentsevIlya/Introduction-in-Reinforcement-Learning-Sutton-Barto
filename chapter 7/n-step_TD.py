import numpy as np
import matplotlib.pyplot as plt

states_dict = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5}


def get_reward(state):
    if state < 6:
        return 0
    else:
        return 1


def online_n_step_TD_0(v, n_step, alpha, gamma=1, s=10):
    states_seen = [s]
    rewards = [0]
    time = 0
    T = float('inf')
    while True:
        time += 1

        if time < T:
            action = np.random.choice([-1, 1])
            new_s = s + action

            reward = 0
            if new_s == 0:
                reward = -1
            elif new_s == states_num + 1:
                reward = 1

            states_seen.append(new_s)
            rewards.append(reward)

            if new_s in [0, states_num + 1]:
                T = time

        update_time = time - n_step
        if update_time >= 0:
            R = 0
            for t in range(update_time + 1, min(update_time + n_step, T) + 1):
                R += gamma**(t - update_time - 1) * rewards[t]

            if update_time + n_step <= T:
                R += gamma**(n_step) * v[states_seen[update_time + n_step]]

            update_state = states_seen[update_time]
            if update_state not in [0, states_num + 1]:
                v[update_state] += alpha * (R - v[update_state])

        if update_time == T - 1:
            break
        s = new_s
    return v


def offline_n_step_TD_0(v, n_step, alpha, gamma=1, s=10):
    sum_delta_v = np.zeros(states_num + 2)
    states_seen = [s]
    rewards = [0]
    time = 0
    T = float('inf')
    while True:
        time += 1

        if time < T:
            action = np.random.choice([-1, 1])
            new_s = s + action

            reward = 0
            if new_s == 0:
                reward = -1
            elif new_s == states_num + 1:
                reward = 1

            states_seen.append(new_s)
            rewards.append(reward)

            if new_s in [0, states_num + 1]:
                T = time

        update_time = time - n_step
        if update_time >= 0:
            R = 0
            for t in range(update_time + 1, min(update_time + n_step, T) + 1):
                R += gamma**(t - update_time - 1) * rewards[t]

            if update_time + n_step <= T:
                R += gamma**(n_step) * v[states_seen[update_time + n_step]]

            update_state = states_seen[update_time]
            if update_state not in [0, states_num + 1]:
                sum_delta_v[update_state] += alpha * (R - v[update_state])

        if update_time == T - 1:
            break
        s = new_s
    return sum_delta_v


def RMSError_online(fig_num, alphas, v_true, states, runs_num, episodes_num,
                    n_step):
    v = np.zeros(states_num + 2)
    plt.figure(fig_num)
    rmse = np.zeros(episodes_num)
    aver_rmse = []
    for alpha in alphas:
        print('alpha = %s' % alpha)
        for run in range(runs_num):
            v_temp = np.copy(v)
            for episode in range(episodes_num):
                v_temp = online_n_step_TD_0(v_temp, n_step, alpha)
                rmse[episode] += np.sqrt(np.sum(np.power(v_true[1:-1] -
                                                v_temp[1:-1], 2)) / states_num)
        rmse = np.array(rmse) / runs_num
        aver_rmse.append(min(np.mean(rmse), 0.55))
    plt.plot(alphas, aver_rmse,
             label='n = ' + str(n_step))
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.xlabel('alpha')
    plt.ylabel('RMSE')


def RMSError_offline(fig_num, alphas, v_true, states, runs_num, episodes_num,
                     n_step):
    v = np.zeros(states_num + 2)
    plt.figure(fig_num)
    rmse = np.zeros(episodes_num)
    aver_rmse = []
    for alpha in alphas:
        print('alpha = %s' % alpha)
        for run in range(runs_num):
            v_temp = np.copy(v)
            for episode in range(episodes_num):
                sum_delta_v = offline_n_step_TD_0(v_temp, n_step, alpha)
                v_temp += sum_delta_v
                rmse[episode] += np.sqrt(np.sum(np.power(v_true[1:-1] -
                                                v_temp[1:-1], 2)) / states_num)
        rmse = np.array(rmse) / runs_num
        aver_rmse.append(min(np.mean(rmse), 0.55))
    print(v_temp)
    plt.plot(alphas, aver_rmse,
             label='n = ' + str(n_step))
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.xlabel('alpha')
    plt.ylabel('RMSE')

states_num = 19
states = np.array(range(1, states_num))
v_true = np.array(range(-20, 22, 2)) / (states_num + 1)
v_true[0] = v_true[-1] = 0

alphas = np.arange(0, 0.33, 0.03)
n_steps = [1, 2, 3, 4, 6, 8, 15, 30, 60, 100, 200, 1000]
#, 3, 5, 8, 15, 30, 60, 100, 200, 1000
for n_step in n_steps:
    print('n = %s' % n_step)
    RMSError_offline(1, alphas, v_true, states,
             runs_num=100,
             episodes_num=10,
             n_step=n_step)

