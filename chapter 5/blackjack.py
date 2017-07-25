import numpy as np
import matplotlib.pyplot as plt


def count_cards_sum(cards_array):
    cards = cards_array.copy()
    cards_sum = sum(cards)
    while cards_sum > 21:
        if 11 not in cards:
            break
        else:
            cards_sum -= 10
            cards.remove(11)
    return cards_sum


def is_there_usable_ace(cards_array):
    cards = cards_array.copy()
    if 11 in cards:
        while 11 in cards:
            cards.remove(11)
        if count_cards_sum(cards) < 11:
            return True
    return False


def get_player_cards():
    value = np.random.randint(12, 22)
    if value == 21:
        return [11, 10]
    if value == 12:
        return [11, 11]
    if np.random.random() > 0.5:
        return [11, value - 11]
    else:
        return [11, 5, value - 6]


class State:
    def __init__(self, dealer_card, card_sum, usable_ace):
        self.dealer_card = dealer_card
        self.card_sum = card_sum
        self.usable_ace = usable_ace

    def __str__(self):
        return "dealer card: " + str(self.dealer_card) + "\n" + \
            "card sum: " + str(self.card_sum) + "\n" + \
            "usable ace: " + str(self.usable_ace) + "\n"

    def to_tuple(self):
        return (self.dealer_card, self.card_sum, self.usable_ace)


class State_Action:
    def __init__(self, state, action):
        self.state = state
        self.action = action

    def __str__(self):
        return str(self.state) + ', ' + self.action

    def to_tuple(self):
        return (self.state.to_tuple(), self.action)


class Dealer:
    def __init__(self):
        self.cards = [np.random.randint(2, 12), np.random.randint(2, 12)]
        self.natural = False
        if count_cards_sum(self.cards) == 21:
            self.natural = True

    def play(self):
        while count_cards_sum(self.cards) < 17:
            self.cards.append(np.random.randint(2, 12))


class Player:
    Q = {}
    Returns = {}
    Visits = {}

    def __init__(self, dealer_card):
        self.cards = get_player_cards()
        self.dealer_card = dealer_card
        self.natural = False
        if count_cards_sum(self.cards) == 21:
            self.natural = True

    def get_state(self):
        return State(self.dealer_card,
                     count_cards_sum(self.cards),
                     is_there_usable_ace(self.cards))

    def get_greedy_action(self):
        state = self.get_state()
        stick_val = self.Q.get(State_Action(state, 'stick').to_tuple())
        hit_val = self.Q.get(State_Action(state, 'hit').to_tuple())
        if stick_val is None and hit_val is None or stick_val == hit_val:
            if state.card_sum < 20:
                return 'hit'
            else:
                return 'stick'
        if stick_val >= hit_val:
            return 'stick'
        else:
            return 'hit'

    def make_action(self):
        g_action = self.get_greedy_action()
        state = self.get_state()
        if state.card_sum >= 21:
            return 'stick'
        if g_action == 'hit':
            action = 'stick'
        else:
            action = 'hit'
        if np.random.random() > 0.5:
            return g_action
        else:
            return action
#        if self.get_state().card_sum > 19:
#            return 'stick'
#        elif np.random.random() > 0.5:
#            return 'hit'
#        else:
#            return 'stick'

    def change_policy(self, states_seen, reward):
        for state in states_seen:
            s_t = state.to_tuple()
            self.Returns[s_t] += reward
            self.Visits[s_t] += 1
            self.Q[s_t] = self.Returns[s_t] / \
                self.Visits[s_t]

    def play(self):
        action = np.random.choice(['hit', 'stick'])
        if count_cards_sum(self.cards) == 21:
            action = 'stick'
        states_seen = []
        while action == 'hit':
            states_seen.append(State_Action(self.get_state(), 'hit'))
            new_card = np.random.randint(2, 12)
            self.cards.append(new_card)
            action = self.make_action()
        if count_cards_sum(self.cards) < 22:
            states_seen.append(State_Action(self.get_state(), 'stick'))
        return states_seen

    def __str__(self):
        return str(self.get_state()) + \
            'player cards: ' + str(self.cards)


class Game:
    def __init__(self):
        self.dealer = Dealer()
        self.player = Player(self.dealer.cards[0])

    def get_reward(self):
        player_sum = count_cards_sum(self.player.cards)
        dealer_sum = count_cards_sum(self.dealer.cards)
#        if self.player.natural:
#            if self.dealer.natural:
#                reward = 0
#            else:
#                reward = 1
#        elif self.dealer.natural:
#            reward = -1
        if player_sum > 21:
            reward = -1
        elif dealer_sum > 21:
            reward = 1
        elif player_sum > dealer_sum:
            reward = 1
        elif player_sum < dealer_sum:
            reward = -1
        else:
            reward = 0
        return reward

    def start(self):
        states_seen = self.player.play()
        self.dealer.play()
        reward = self.get_reward()

        self.player.change_policy(states_seen, reward)
#        for state in states_seen:
#            print(state)
#        print(reward)
#        print(self.dealer.cards)
#        a = input('a')

n_games = 500000
games = {}
returns = {}
visits = {}

Q = {}
for i in range(2, 12):
    for j in range(12, 22):
        for ace in [True, False]:
            for act in ['hit', 'stick']:
                Q[((i, j, ace), act)] = 0
                returns[((i, j, ace), act)] = 0
                visits[((i, j, ace), act)] = 0

for i in range(n_games):
    if i % 10000 == 0:
        print(i)
    Player.Q = Q
    Player.Returns = returns
    Player.Visits = visits
    game = Game()
    game.start()

Q = Player.Q
print('Ok')
x_usable_ace = []
y_usable_ace = []
val_usable_ace = []

x = []
y = []
val = []

for i in range(2, 12):
    for j in range(12, 22):
        x_usable_ace.append(i)
        y_usable_ace.append(j)
        x.append(i)
        y.append(j)
        q_h = Q.get(((i, j, True), 'hit'))
        if q_h is None:
            q_h = -1
        q_s = Q.get(((i, j, True), 'stick'))
        if q_s is None:
            q_s = -1
        if q_h >= q_s:
            val_usable_ace.append(1)
        else:
            val_usable_ace.append(0)

        q_h = Q.get(((i, j, False), 'hit'))
        if q_h is None:
            q_h = -1
        q_s = Q.get(((i, j, False), 'stick'))
        if q_s is None:
            q_s = -1
        if q_h >= q_s:
            val.append(1)
        else:
            val.append(0)

plt.figure(1)
plt.title("No usable ace")
for i in range(len(val)):
    if val[i] == 0:
        plt.plot(x[i], y[i], 'ro')
    else:
        plt.plot(x[i], y[i], 'go')
plt.show()

plt.figure(2)
plt.title("Usable ace")
for i in range(len(val_usable_ace)):
    if val_usable_ace[i] == 0:
        plt.plot(x_usable_ace[i], y_usable_ace[i], 'ro')
    else:
        plt.plot(x_usable_ace[i], y_usable_ace[i], 'go')
plt.show()


