import numpy as np
import matplotlib.pyplot as plt


def lambda_return(v, alpha, gamma=1, lambda_=0.5, s=10):
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
                break
        s = new_s

    for update_time in range(0, T):
        R_lambda = 0
        update_state = states_seen[update_time]
        if update_state not in [0, states_num + 1]:
            for n in range(1, T - update_time):
                R = 0
                for t in range(update_time + 1, min(update_time + n, T) + 1):
                    R += gamma**(t - update_time - 1) * rewards[t]

                if update_time + n <= T:
                    R += gamma**(n) * v[states_seen[update_time + n]]

                R_lambda += lambda_**(n - 1) * R
            R_lambda *= (1 - lambda_)
            R_lambda += lambda_**(T - update_time - 1) * rewards[-1]
            sum_delta_v[update_state] += alpha * (R_lambda - v[update_state])
    return sum_delta_v


def online_TD_lambda(v, alpha, gamma=1, lambda_=0.5, s=10):
    e = np.zeros(states_num + 2)
    while s not in [0, states_num + 1]:
        action = np.random.choice([-1, 1])
        new_s = s + action

        reward = states_reward[new_s]

        delta = reward + gamma * v[new_s] - v[s]
        e[s] += 1

        v[1: states_num + 1] += alpha * delta * e[1: states_num + 1]
        e[1: states_num + 1] *= gamma * lambda_

        s = new_s
    return v


def online_TD_lambda_replacing_trace(v, alpha, gamma=1, lambda_=0.5, s=10):
    e = np.zeros(states_num + 2)
    while s not in [0, states_num + 1]:
        action = np.random.choice([-1, 1])
        new_s = s + action

        reward = states_reward[new_s]

        delta = reward + gamma * v[new_s] - v[s]

        v[1: states_num + 1] += alpha * delta * e[1: states_num + 1]
        e[1: states_num + 1] *= gamma * lambda_
        e[s] = 1

        s = new_s
    return v


def RMSError_online(fig_num, alphas, v_true, states, runs_num, episodes_num,
                    lambda_, method):
    v = np.zeros(states_num + 2)
    plt.figure(fig_num)
    rmse = np.zeros(episodes_num)
    aver_rmse = []
    for alpha in alphas:
        print('alpha = %s' % alpha)
        for run in range(runs_num):
            v_temp = np.copy(v)
            for episode in range(episodes_num):
                v_temp = \
                    method(v=v_temp,
                           alpha=alpha,
                           gamma=1,
                           lambda_=lambda_)
                rmse[episode] += np.sqrt(np.sum(np.power(v_true[1:-1] -
                                                v_temp[1:-1], 2)) / states_num)
        rmse = np.array(rmse) / runs_num
        aver_rmse.append(min(np.mean(rmse), 0.55))
    plt.plot(alphas, aver_rmse,
             label='lambda = ' + str(lambda_))
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.xlabel('alpha')
    plt.ylabel('RMSE')


def RMSError_offline(fig_num, alphas, lambda_, v_true, states, runs_num,
                     episodes_num):
    v = np.zeros(states_num + 2)
    plt.figure(fig_num)
    rmse = np.zeros(episodes_num)
    aver_rmse = []
    for alpha in alphas:
        print('alpha = %s' % alpha)
        for run in range(runs_num):
            print('Run: %s' % run)
            v_temp = np.copy(v)
            for episode in range(episodes_num):
                sum_delta_v = lambda_return(v=v_temp, alpha=alpha, gamma=1,
                                            lambda_=lambda_)
                v_temp += sum_delta_v
                rmse[episode] += np.sqrt(np.sum(np.power(v_true[1:-1] -
                                                v_temp[1:-1], 2)) / states_num)
        rmse = np.array(rmse) / runs_num
        aver_rmse.append(min(np.mean(rmse), 0.55))
        print(v_temp)
    plt.plot(alphas, aver_rmse,
             label='lambda = ' + str(lambda_))
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.xlabel('alpha')
    plt.ylabel('RMSE')


def RMSError_online_lambda(fig_num, alpha, v_true, states, runs_num, episodes_num,
                    lambdas, method):
    v = np.zeros(states_num + 2)
    plt.figure(fig_num)
    rmse = np.zeros(episodes_num)
    aver_rmse = []
    for lambda_ in lambdas:
        print('lambda = %s' % lambda_)
        for run in range(runs_num):
            v_temp = np.copy(v)
            for episode in range(episodes_num):
                v_temp = \
                    method(v=v_temp,
                           alpha=alpha,
                           gamma=1,
                           lambda_=lambda_)
                rmse[episode] += np.sqrt(np.sum(np.power(v_true[1:-1] -
                                                v_temp[1:-1], 2)) / states_num)
        rmse = np.array(rmse) / runs_num
        aver_rmse.append(min(np.mean(rmse), 0.55))
    plt.plot(lambdas, aver_rmse,
             label='alpha = ' + str(alpha))
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.xlabel('lambda')
    plt.ylabel('RMSE')

states_num = 19
states_reward = np.zeros(states_num + 2)
states_reward[0] = -1
states_reward[-1] = 1
states = np.array(range(1, states_num))
v_true = np.array(range(-20, 22, 2)) / (states_num + 1)
v_true[0] = v_true[-1] = 0


alphas = np.arange(0, 1.1, 0.1)
lambdas = [0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.975, 0.99, 1]
#, 3, 5, 8, 15, 30, 60, 100, 200, 1000
RMSError_online_lambda(fig_num=1,
                       alpha=0.6,
                       lambdas=lambdas,
                       v_true=v_true,
                       states=states,
                       runs_num=100,
                       episodes_num=10,
                       method=online_TD_lambda)
RMSError_online_lambda(fig_num=1,
                       alpha=0.2,
                       lambdas=lambdas,
                       v_true=v_true,
                       states=states,
                       runs_num=100,
                       episodes_num=10,
                       method=online_TD_lambda)

#for lambda_ in lambdas:
#    print('lambda = %s' % lambda_)
#    RMSError_online(fig_num=1,
#                       alphas=alphas,
#                       lambda_=lambda_,
#                       v_true=v_true,
#                       states=states,
#                       runs_num=100,
#                       episodes_num=10,
#                       method=online_TD_lambda_replacing_trace)
