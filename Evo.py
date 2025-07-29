import random
import numpy as np
import pickle
import os
from collections import defaultdict

class GoFishEnv:
    def DefaultQ(self):
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    def __init__(self):
        self.CONST_DECK = {13 : 4, 12 : 4, 11 : 4, 10 : 4, 9 : 4, 8 : 4, 7 : 4, 6 : 4, 5 : 4, 4 : 4, 3 : 4, 2 : 4, 1 : 4}
        self.Cards = {13 : 4, 12 : 4, 11 : 4, 10 : 4, 9 : 4, 8 : 4, 7 : 4, 6 : 4, 5 : 4, 4 : 4, 3 : 4, 2 : 4, 1 : 4}
        self.Hand1 = []
        self.Hand2 = []
        self.Score1 = 0
        self.Score2 = 0
        self.potential1 = {13 : 0, 12 : 0, 11 : 0, 10 : 0, 9 : 0, 8 : 0, 7 : 0, 6 : 0, 5 : 0, 4 : 0, 3 : 0, 2 : 0, 1 : 0}
        self.potential2 = {13 : 0, 12 : 0, 11 : 0, 10 : 0, 9 : 0, 8 : 0, 7 : 0, 6 : 0, 5 : 0, 4 : 0, 3 : 0, 2 : 0, 1 : 0}
        self.Q = defaultdict(self.DefaultQ)
        if os.path.exists("q_table.pkl"):
            with open("q_table.pkl", "rb") as f:
                self.Q = defaultdict(self.DefaultQ, pickle.load(f))
            print(self.Q)

    def Reset(self):
        self.Cards = self.CONST_DECK.copy()
        self.Hand1 = []
        self.Hand2 = []
        self.Score1 = 0
        self.Score2 = 0
        self.potential1 = {13: 0, 12: 0, 11: 0, 10: 0, 9: 0, 8: 0, 7: 0, 6: 0, 5: 0, 4: 0, 3: 0, 2: 0, 1: 0}
        self.potential2 = {13: 0, 12: 0, 11: 0, 10: 0, 9: 0, 8: 0, 7: 0, 6: 0, 5: 0, 4: 0, 3: 0, 2: 0, 1: 0}

    def Deal(self):
        self.Reset()
        while len(self.Hand1) < 7:
            self.Draw(self.Hand1)
        while len(self.Hand2) < 7:
            self.Draw(self.Hand2)

    def Draw(self, player):
        if self.Cards:
            R = random.choice(list(self.Cards.keys()))
            player.append(R)
            self.Cards[R] -= 1
            if self.Cards[R] <= 0:
                del self.Cards[R]

    def Question(self, card, player, victim):
        if card in victim and card in player:
            while card in victim:
                player.append(card)
                victim.remove(card)
            return True
        return False

    def CheckBooks(self, hand):
        books = 0
        for card in set(hand):
            if hand.count(card) == 4:
                books += 1
                del self.potential1[card]
                del self.potential2[card]
                for i in range(4):
                    hand.remove(card)
        return books

    def GetState(self, player, opponent, potential, playerscore, opponentscore):
        useful = potential.copy()
        for card in list(useful.keys()):
            if card not in player:
                del useful[card]
        if len(useful) == 0:
            return (frozenset(player), 0, len(opponent))
        return (frozenset(player), max(useful, key = useful.get), len(opponent))

alpha = 0.1
gamma = 0.7
epsilon = 1
e_decay = 0.99999995

env = GoFishEnv()

def GetAction(env, state, epsilon):
    if random.random() < epsilon and len(list(state[0])):
        return random.choice(list(state[0]))
    else:
        for i in range(13):
            if not i + 1 in list(state[0]):
                env.Q[state][int(i)] = -9999
        return int(np.argmax(env.Q[state]) + 1)

summary = 75000
Save_Interval = 200000
win = 0
loss = 0

for i in range(15000000):
    states1 = []
    action1 = []
    states2 = []
    action2 = []
    if i == Save_Interval and i > 0:
        print("Saving: DO NOT SHUT OFF")
        with open("q_table.pkl", "wb") as f:
            pickle.dump(env.Q, f)
        print(f"Q-table auto-saved at episode {i}")
        Save_Interval += 150000
    if i == summary:
        print(len(env.Q), win/(loss + win), epsilon)
        win = 0
        loss = 0
        summary += 75000
    env.Deal()
    over = False
    while not over:
        P1 = True
        P2 = True
        while P1:
            if len(env.Hand1) == 0:
                env.Draw(env.Hand1)
            if len(env.Hand2) == 0:
                env.Draw(env.Hand2)
            state = env.GetState(env.Hand1, env.Hand2, env.potential1, env.Score1, env.Score2)
            action = GetAction(env, state, epsilon)
            states1.append(state)
            action1.append(action)

            won = env.Question(action, env.Hand1, env.Hand2)
            if action in env.Hand1:
                env.potential2[action] += 1
                env.potential1[action] = 0
            else:
                env.Q[state][action - 1] = -9
            reward = -1
            if won:
                reward = 1
                env.Score1 += env.CheckBooks(env.Hand1)
                win += 1

            newstate = env.GetState(env.Hand1, env.Hand2, env.potential1, env.Score1, env.Score2)
            bestfuture = np.max(env.Q[newstate])

            env.Q[state][action - 1] += alpha * (reward + gamma * bestfuture - env.Q[state][action - 1])

            if not won:
                loss += 1
                if len(env.Cards) > 0:
                    env.Draw(env.Hand1)
                    while len(env.Hand1) == 0:
                        env.Draw(env.Hand1)
                    env.Score1 += env.CheckBooks(env.Hand1)
                P1 = False
        while P2:
            if len(env.Hand2) == 0:
                env.Draw(env.Hand2)
            if len(env.Hand1) == 0:
                env.Draw(env.Hand1)
            state = env.GetState(env.Hand2, env.Hand1, env.potential2, env.Score2, env.Score1)
            action = GetAction(env, state, epsilon)
            states2.append(state)
            action2.append(action)

            won = env.Question(action, env.Hand2, env.Hand1)
            if action in env.Hand2:
                env.potential1[action] += 1
                env.potential2[action] = 0
            else:
                env.Q[state][action - 1] = -9
            reward = -1
            if won:
                reward = 1
                env.Score2 += env.CheckBooks(env.Hand2)
                win += 1

            newstate = env.GetState(env.Hand2, env.Hand1, env.potential2, env.Score2, env.Score1)
            bestfuture = np.max(env.Q[newstate])

            env.Q[state][action - 1] += alpha * (reward + gamma * bestfuture - env.Q[state][action - 1])

            if not won:
                loss += 1
                if len(env.Cards) > 0:
                    env.Draw(env.Hand2)
                    while len(env.Hand2) == 0:
                        env.Draw(env.Hand2)
                    env.Score2 += env.CheckBooks(env.Hand2)
                P2 = False
        if not env.Cards and not env.Hand1 and not env.Hand2:
            if env.Score1 < env.Score2:
                for i in range(len(states1)):
                    reward = -((env.Score2 + 1) / (env.Score1 + 1))
                    env.Q[states1[i]][action1[i] - 1] += alpha * (reward - env.Q[states1[i]][action1[i] - 1])

                for i in range(len(states2)):
                    reward = 1 + ((env.Score1 + 1) / (env.Score2 + 1))
                    env.Q[states2[i]][action2[i] - 1] += alpha * (reward - env.Q[states2[i]][action2[i] - 1])
            else:
                for i in range(len(states2)):
                    reward = -((env.Score1 + 1) / (env.Score2 + 1))
                    env.Q[states2[i]][action2[i] - 1] += alpha * (reward - env.Q[states2[i]][action2[i] - 1])

                for i in range(len(states1)):
                    reward = 1 + ((env.Score2 + 1) / (env.Score1 + 1))
                    env.Q[states1[i]][action1[i] - 1] += alpha * (reward - env.Q[states1[i]][action1[i] - 1])

            if epsilon > 0.05:
                epsilon *= e_decay
            over = True

print("Saving Q-table...")
with open("q_table.pkl", "wb") as f:
    pickle.dump(env.Q, f)
    print("Q-table saved!")