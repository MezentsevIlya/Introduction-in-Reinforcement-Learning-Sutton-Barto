import matplotlib.pyplot as plt
import random
import numpy as np
import math

class Agent:
    def __init__(self, n, start_values = 0.0, alpha = 0.1):
        self.n = n
        self.actions_values = np.array([start_values] * n)
        self.actions_chosen_count = np.zeros(n)
        self.alpha = alpha
        self.is_best_action_array = []
        
    def get_actions_values(self):
        return self.actions_values
        
    def make_action(self):
        return random.randint(0, self.n - 1)
            
    def change_action_value(self, action, value):
        self.actions_chosen_count[action] += 1
        act_count = self.actions_chosen_count[action]
        if self.alpha == "adaptive":
            alpha = 1 / act_count
        else:
            alpha = self.alpha
        self.actions_values[action] += \
            alpha * (value - self.actions_values[action])
        

class EpsGreedy_Agent(Agent):
    def __init__(self, n, start_values = 0.0, alpha = 0.1, epsilon = 0.0):
        Agent.__init__(self, n, start_values, alpha)
        self.epsilon = epsilon
        
    
    def make_action(self):
        if random.random() >= self.epsilon * (self.n - 1) / self.n:
            return np.argmax(self.actions_values)
        else:            
            return random.randint(0, self.n - 1)
            
class Softmax_Agent(Agent):
    def __init__(self, n, start_values = 0.0, alpha = 0.1, tau = 0.0):
        Agent.__init__(self, n, start_values, alpha)
        self.tau = tau
        self.actions_prob = np.ones(n) / n
    
    def make_action(self):
        return np.random.choice(a = np.arange(self.n), p = self.actions_prob)
    
    def change_action_value(self, action, value):
        self.actions_chosen_count[action] += 1
        act_count = self.actions_chosen_count[action]
        if self.alpha == "adaptive":
            alpha = 1 / act_count
        else:
            alpha = self.alpha
        self.actions_values[action] += \
            alpha * (value - self.actions_values[action])
        new_value = self.actions_values[action]
        self.change_prob(new_value, action)
            
    def change_prob(self, new_value, action):
        denominator = np.sum(np.exp(self.actions_values / self.tau))
        self.actions_prob = np.exp(self.actions_values / self.tau) / denominator
        
            
class Reinforcement_Comparison_Agent(Agent):
    def __init__(self, n, start_values = 0.0, alpha = 0.1, beta = 0.1, r_reward = 1.0):
        Agent.__init__(self, n, start_values, alpha)
        self.preferences = np.array([0.0] * n)
        self.actions_prob = np.ones(n) / n
        self.beta = beta
        self.r_reward = r_reward
        
    def make_action(self):
        return np.random.choice(a = np.arange(self.n), p = self.actions_prob)
    
    def change_action_value(self, action, value):        
        self.change_prob(action, value)
        
    def change_prob(self, action, value):
        self.preferences[action] = \
            self.preferences[action] + self.beta * (value - self.r_reward)

        denominator = np.sum([np.exp(p) for p in self.preferences])
        self.actions_prob = \
            np.array([np.exp(p) for p in self.preferences]) / denominator
        self.r_reward += self.alpha * (value - self.r_reward) 
        
class Pursuit_Agent(Agent):
    def __init__(self, n, start_values = 0.0, alpha = 0.1, beta = 0.1):
        Agent.__init__(self, n, start_values, alpha)
        self.actions_prob = np.ones(n) / n
        self.beta = beta
    
    def get_greedy_action(self):
        return np.argmax(self.actions_values)
    
    def make_action(self):
        return np.random.choice(a = np.arange(self.n), p = self.actions_prob)
    
    def change_action_value(self, action, value):
        self.actions_chosen_count[action] += 1
        act_count = self.actions_chosen_count[action]
        if self.alpha == "adaptive":
            alpha = 1 / act_count
        else:
            alpha = self.alpha
        self.actions_values[action] += \
            alpha * (value - self.actions_values[action])
        self.change_prob()
            
    def change_prob(self):
        greedy_action = self.get_greedy_action()
        
        self.actions_prob += self.beta * (-self.actions_prob)
        self.actions_prob[greedy_action] = \
            1 - (sum(self.actions_prob) - self.actions_prob[greedy_action])
#        print(self.actions_prob[greedy_action])
    

class Environment:
    def __init__(self, n):
        self.n = n
        self.actions_values = np.zeros(n)
        
    def get_actions_values(self):
        return self.actions_values
    
    def get_some_value(self, action):
        return random.normalvariate(self.actions_values[action], 1)
        
    def is_best_action(self, action):
        return action == self.best_action
        
class Stable_Environment(Environment):
    def __init__(self, n):
        Environment.__init__(self, n)
        for i in range(n):
            self.actions_values[i] = random.normalvariate(0, 1)
        self.best_action = np.argmax(self.actions_values)
        
        
class Unstable_Environment(Environment):
    def __init__(self, n):
        Environment.__init__(self, n)
        for i in range(n):
            self.actions_values[i] = 0.5
        self.best_action = 0
    
    def change_values(self):
        for i in range(len(self.actions_values)):
            self.actions_values[i] += random.normalvariate(0, math.sqrt(0.01))
        self.best_action = \
            np.argmax(self.actions_values)
        
    def get_some_value(self, action):
        return random.normalvariate(self.actions_values[action], 1)
    
    def is_best_action(self, action):        
        self.change_values()
        return action == self.best_action
        
        
class Game:
    def __init__(self, agent, env, n_games):
        self.agent = agent
        self.env = env
        self.n_games = n_games
        self.games_played = 0
        self.best_actions_count = 0
        self.percent_best_actions = np.zeros(n_games)
        self.values = np.zeros(n_games)
        
    def make_action(self):
        action = self.agent.make_action()
        value = self.env.get_some_value(action)
        self.agent.change_action_value(action, value)
        return (action, value)
        
    def check_action(self, action):
        is_best = 0.0
        if self.env.is_best_action(action):
            self.best_actions_count += 1
            is_best = 1.0
        self.agent.is_best_action_array.append(is_best)
        self.percent_best_actions[self.games_played - 1] = \
            self.best_actions_count / self.games_played
        
        
    def play(self):
        action, value = self.make_action()
        self.games_played += 1
        self.check_action(action)
        self.values[self.games_played - 1] = value
        
    def start(self):
        for i in range(self.n_games):
            self.play()
            

            
def make_experiment_greedy_agent(n_games,
                                 n_hand,
                                 n_experiments,
                                 epsilon_array,
                                 alpha = 0.1,
                                 start_values = 0.0,
                                 env_stable = True):
    pba = []
    av_v = []
    b_a = []
    
    for epsilon in epsilon_array:
        percent_best_actions = np.zeros(n_games)
        av_values = np.zeros(n_games)
        b_act = np.zeros(n_games)
        for i in range(n_experiments):
            if i % 100 == 0:
                print(i)
            agent = EpsGreedy_Agent(n_hand, start_values, alpha, epsilon)
            if env_stable:
                env = Stable_Environment(n_hand)
            else:
                env = Unstable_Environment(n_hand)
                    
            game = Game(agent, env, n_games)
            game.start()
                        
            percent_best_actions += game.percent_best_actions
            av_values += game.values
            b_act += np.array(game.agent.is_best_action_array)
        
        pba.append(percent_best_actions / n_experiments * 100)
        av_v.append(av_values / n_experiments)
        b_a.append(b_act / n_experiments * 100)
        
    return (pba, av_v, b_a)
    
    
def make_soft_max_experiment(n_games,
                             n_hand,
                             n_experiments,
                             tau_array,
                             alpha = 0.1,
                             start_values = 0.0):
    pba = []
    av_v = []
    b_a = []
    for tau in tau_array:
        percent_best_actions = np.zeros(n_games)
        av_values = np.zeros(n_games)
        b_act = np.zeros(n_games)
        for i in range(n_experiments):
            if i % 10 == 0:
                    print(i)
            agent = Softmax_Agent(n_hand, start_values, alpha, tau)
            env = Stable_Environment(n_hand)
                        
            game = Game(agent, env, n_games)
            game.start()
                
            percent_best_actions += game.percent_best_actions
            av_values += game.values
            b_act += np.array(game.agent.is_best_action_array)
            
        pba.append(percent_best_actions / n_experiments * 100)
        av_v.append(av_values / n_experiments)
        b_a.append(b_act / n_experiments * 100)
        
    return (pba, av_v, b_a)
    
def make_experiment_rc_agent(n_games,
                             n_hand,
                             n_experiments,
                             alpha,
                             beta_array,
                             r_reward,
                             start_values = 0.0,
                             env_stable = True):
    pba = []
    av_v = []
    b_a = []
    for beta in beta_array:
        percent_best_actions = np.zeros(n_games)
        av_values = np.zeros(n_games)
        b_act = np.zeros(n_games)
        for i in range(n_experiments):
            if i % 10 == 0:
                print(i)
            agent = Reinforcement_Comparison_Agent(n_hand,
                                                   start_values,
                                                   alpha,
                                                   beta,
                                                   r_reward)
            if env_stable:
                env = Stable_Environment(n_hand)
            else:
                env = Unstable_Environment(n_hand)
                    
            game = Game(agent, env, n_games)
            game.start()
            
            percent_best_actions += game.percent_best_actions
            av_values += game.values
            b_act += np.array(game.agent.is_best_action_array)
        
        pba.append(percent_best_actions / n_experiments * 100)
        av_v.append(av_values / n_experiments)
        b_a.append(b_act / n_experiments * 100)
        
    return (pba, av_v, b_a)
    
def make_experiment_pursuit_agent(n_games,
                             n_hand,
                             n_experiments,
                             alpha,
                             beta_array,
                             start_values = 0.0,
                             env_stable = True):
    pba = []
    av_v = []
    b_a = []
    for beta in beta_array:
        percent_best_actions = np.zeros(n_games)
        av_values = np.zeros(n_games)
        b_act = np.zeros(n_games)
        for i in range(n_experiments):
            if i % 10 == 0:
                print(i)
            agent = Pursuit_Agent(n_hand,
                                  start_values,
                                  alpha,
                                  beta)
            if env_stable:
                env = Stable_Environment(n_hand)
            else:
                env = Unstable_Environment(n_hand)
                    
            game = Game(agent, env, n_games)
            game.start()
            
            percent_best_actions += game.percent_best_actions
            av_values += game.values
            b_act += np.array(game.agent.is_best_action_array)
        
        pba.append(percent_best_actions / n_experiments * 100)
        av_v.append(av_values / n_experiments)
        b_a.append(b_act / n_experiments * 100)
        
    return (pba, av_v, b_a)

        
def plot_results(pba, av_v, b_a, label):
    plt.figure(1)
    plt.plot(pba,
             linewidth=2,
             label=r'$%s = %s$' % (label[0], label[1]))
    plt.xlabel('Games')
    plt.ylabel('%\nOptimal\naction', rotation=0)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.6),
               ncol=2, fancybox=True, shadow=True, prop={'size':16})
    plt.figure(2)
    plt.plot(av_v,
             linewidth=1,
             label=r'$%s = %s$' % (label[0], label[1]))
    plt.xlabel('Games')
    plt.ylabel('Average reward')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.6),
               ncol=2, fancybox=True, shadow=True, prop={'size':16})
    plt.figure(3)
    plt.plot(b_a,
             linewidth=2,
             label=r'$%s = %s$' % (label[0], label[1]))
    plt.xlabel('Games')
    plt.ylabel('%\nAgent\nchosen\noptimal')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.6),
               ncol=2, fancybox=True, shadow=True, prop={'size':16})
    
def plot_from_array(pba_array, av_v_array, is_best_act_array, label):
    for i in range(len(pba_array)):
        plot_results(pba_array[i],
                     av_v_array[i],
                     is_best_act_array[i],
                     (label[0], label[1][i]))
    
n_games = 1000
n_hand = 10
n_experiments = 2000
epsilon_array = [0.1]

(pba_array, av_v_array, is_best_act_array) = make_experiment_greedy_agent(n_games,
                                                       n_hand,
                                                       n_experiments,
                                                       epsilon_array,
                                                       alpha = 'adaptive',
#                                                       alpha = 0.1,
                                                       start_values = 5.0,
                                                       env_stable = True)

plot_from_array(pba_array,
                av_v_array,
                is_best_act_array,
                (r'\epsilon-greedy, \epsilon', epsilon_array))

#best tau = 0.2
#tau = [0.2, 0.5]
#(pba_array, av_v_array, is_best_act_array) = make_soft_max_experiment(n_games,
#                                       n_hand,
#                                       n_experiments,
#                                       tau,
#                                       start_values = 0.0,
#                                       alpha = "adaptive")
#plot_from_array(pba_array, av_v_array, is_best_act_array, (r'softmax, \tau', tau))



###make_experiment_unstable_env(n_games, n_hand, n_experiments)
##
#best: beta = 0.5, alpha = 0.1
beta_array = [0.5]
alpha = 0.1
(pba_array, av_v_array, is_best_act_array) = make_experiment_rc_agent(n_games,
                                                   n_hand,
                                                   n_experiments,
                                                   alpha,
                                                   beta_array,
                                                   r_reward = 1.5,
                                                   start_values = 0.0,
                                                   env_stable = True)

plot_from_array(pba_array,
                av_v_array,
                is_best_act_array,
                (r'r. comparison, \beta', beta_array))
###
beta_array = [0.03, 0.01]
alpha = "adaptive"
(pba_array, av_v_array, is_best_act_array) = make_experiment_pursuit_agent(n_games,
                                                   n_hand,
                                                   n_experiments,
                                                   alpha,
                                                   beta_array,
                                                   start_values = 0.0,
                                                   env_stable = True)

plot_from_array(pba_array, av_v_array, is_best_act_array, (r'pursuit, \beta', beta_array))











