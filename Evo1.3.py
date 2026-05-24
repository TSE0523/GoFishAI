import random
import time

import joblib
import numpy as np
import os
from collections import defaultdict
from collections import OrderedDict
from copy import deepcopy
from joblib import Parallel, delayed
import bisect

starttime = time.time()

def DefaultQ(self):
    return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

GLOBALQ = defaultdict(DefaultQ)
if os.path.exists("q_table.pkl"):
    with open("q_table.pkl", "rb") as f:
        GLOBALQ = deepcopy(defaultdict(DefaultQ, joblib.load("q_table.pkl")))
print(GLOBALQ)

class GoFishEnv:
    def __init__(self):
        self.CONST_DECK = {13 : 4, 12 : 4, 11 : 4, 10 : 4, 9 : 4, 8 : 4, 7 : 4, 6 : 4, 5 : 4, 4 : 4, 3 : 4, 2 : 4, 1 : 4}
        self.Cards = {13 : 4, 12 : 4, 11 : 4, 10 : 4, 9 : 4, 8 : 4, 7 : 4, 6 : 4, 5 : 4, 4 : 4, 3 : 4, 2 : 4, 1 : 4}
        self.Hand1 = []
        self.Hand2 = []
        self.Score1 = 0
        self.Score2 = 0
        self.potential1 = {13 : 0, 12 : 0, 11 : 0, 10 : 0, 9 : 0, 8 : 0, 7 : 0, 6 : 0, 5 : 0, 4 : 0, 3 : 0, 2 : 0, 1 : 0}
        self.potential2 = {13 : 0, 12 : 0, 11 : 0, 10 : 0, 9 : 0, 8 : 0, 7 : 0, 6 : 0, 5 : 0, 4 : 0, 3 : 0, 2 : 0, 1 : 0}
        self.Q = GLOBALQ

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
            cards = list(self.Cards.keys())
            weights = list(self.Cards.values())
            R = random.choices(cards, weights, k=1)[0]
            bisect.insort(player, R)
            self.Cards[R] -= 1
            if self.Cards[R] <= 0:
                del self.Cards[R]

    def Question(self, card, player, victim):
        if card in victim and card in player:
            while card in victim:
                bisect.insort(player, card)
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

    def GetState(self, player, opponent, potential, pastguess):
        length = 2
        if len(opponent) <= 13:
            length = 0
        elif len(opponent) > 13 and len(opponent) < 27:
            length = 1

        hand = 0
        for i in range(1, 14):
            num = player.count(i)
            if num > 1:
                hand |= (1 << (i - 1))

        potential = dict(potential)
        useful = potential.copy()
        for card in list(useful.keys()):
            if card not in player:
                del useful[card]
        if len(useful) == 0:
            return (hand | (pastguess << 13) | (length << 21))

        state = (hand | (pastguess << 13) | ((max(useful, key = useful.get) << 17)) | (length << 21))
        return state

class LRU_CACHE:
    def __init__(self, maxsize=50000):
        self.cache = OrderedDict()
        self.maxsize = maxsize

    def Get(self, state, encoder):
        if state in self.cache:
            self.cache.move_to_end(state)
            return self.cache[state]
        bitstate = encoder(list(state[0]), list(state[1]), dict(state[2]), state[3])
        self.cache[state] = bitstate
        if len(self.cache) > self.maxsize:
            self.cache.popitem(last=False)
        return bitstate

def DefaultQ():
    return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

env = GoFishEnv()
LRU = LRU_CACHE()

def GetAction(env, state, epsilon, hand):
    if np.random.rand() < epsilon and len(hand):
        return np.random.choice(hand)
    else:
        q = env.Q[state].copy()
        for i in range(13):
            if not i + 1 in hand:
                q[i] = -9999
        return np.argmax(q) + 1

def Training(TRIALS, epsilon):
    env = GoFishEnv()
    LRU = LRU_CACHE()
    alpha = 0.4
    gamma = 0.7

    for i in range(TRIALS):
        states1 = []
        action1 = []
        states2 = []
        action2 = []

        env.Deal()
        pastguess1 = 0
        pastguess2 = 0

        P1 = True
        P2 = True
        while env.Score1 + env.Score2 < 13:
            P1 = True
            P2 = True
            while P1:
                if len(env.Hand1) == 0:
                    env.Draw(env.Hand1)
                if len(env.Hand2) == 0:
                    env.Draw(env.Hand2)
                state = LRU.Get((tuple(env.Hand1), tuple(env.Hand2), tuple(sorted(env.potential1.items())), pastguess1), env.GetState)
                action = GetAction(env, state, epsilon, env.Hand1)
                pastguess1 = action
                states1.append(state)
                action1.append(action)

                won = env.Question(action, env.Hand1, env.Hand2)
                if action in env.Hand1:
                    env.potential2[action] += 1
                    env.potential1[action] = 0
                else:
                    env.Q[state][action - 1] -= 0.6
                reward = -1
                if won:
                    points = env.CheckBooks(env.Hand1)
                    reward = 1 + points
                    env.Score1 += points

                newstate = LRU.Get((tuple(env.Hand1), tuple((env.Hand2)), tuple(sorted(env.potential1.items())), pastguess1), env.GetState)
                bestfuture = np.max(env.Q[newstate])

                if env.Q[state][action - 1] < 5:
                    env.Q[state][action - 1] += alpha * (reward + gamma * bestfuture - env.Q[state][action - 1])

                if not won:
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
                state = LRU.Get((tuple((env.Hand2)), tuple((env.Hand1)), tuple(sorted(env.potential2.items())), pastguess2), env.GetState)
                action = GetAction(env, state, epsilon, env.Hand2)
                pastguess2 = action
                states2.append(state)
                action2.append(action)

                won = env.Question(action, env.Hand2, env.Hand1)
                if action in env.Hand2:
                    env.potential1[action] += 1
                    env.potential2[action] = 0
                else:
                    env.Q[state][action - 1] -= 0.6
                reward = -1
                if won:
                    points = env.CheckBooks(env.Hand2)
                    reward = 1 + points
                    env.Score2 += points

                newstate = LRU.Get((tuple((env.Hand2)), tuple((env.Hand1)), tuple(sorted(env.potential2.items())), pastguess2), env.GetState)
                bestfuture = np.max(env.Q[newstate])
                if env.Q[state][action - 1] < 5:
                    env.Q[state][action - 1] += alpha * (reward + gamma * bestfuture - env.Q[state][action - 1])

                if not won:
                    if len(env.Cards) > 0:
                        env.Draw(env.Hand2)
                        while len(env.Hand2) == 0:
                            env.Draw(env.Hand2)
                        env.Score2 += env.CheckBooks(env.Hand2)
                    P2 = False
        if not env.Cards and not env.Hand1 and not env.Hand2:
            if env.Score1 < env.Score2:
                for i in range(len(states1)):
                    if env.Q[states1[i]][action1[i] - 1] > -5:
                        reward = -((env.Score2 + 1) / (env.Score1 + 1))
                        env.Q[states1[i]][action1[i] - 1] += alpha * (reward - env.Q[states1[i]][action1[i] - 1])

                for i in range(len(states2)):
                    if env.Q[states2[i]][action2[i] - 1] < 5:
                        reward = 2 + ((env.Score1 + 1) / (env.Score2 + 1))
                        env.Q[states2[i]][action2[i] - 1] += alpha * (reward - env.Q[states2[i]][action2[i] - 1])
            else:
                for i in range(len(states2)):
                    if env.Q[states2[i]][action2[i] - 1] > -5:
                        reward = -((env.Score1 + 1) / (env.Score2 + 1))
                        env.Q[states2[i]][action2[i] - 1] += alpha * (reward - env.Q[states2[i]][action2[i] - 1])

                for i in range(len(states1)):
                    if env.Q[states1[i]][action1[i] - 1] < 5:
                        reward = 2 + ((env.Score2 + 1) / (env.Score1 + 1))
                        env.Q[states1[i]][action1[i] - 1] += alpha * (reward - env.Q[states1[i]][action1[i] - 1])
    return dict(env.Q)

def Merge(Global, Local, Agents):
    for state, actions in Local.items():
        if not state in Global:
            Global[state] = actions.copy()
        else:
            for i in range(len(actions)):
                Global[state][i] += actions[i]
                Global[state][i] /= 2
    return Global

AUTOSAVE = 100000
TOTAL = 1000000
AGENTS = 2

if __name__ == "__main__":
    progress = 0
    epsilon = 1.0
    e_decay = 0.9
    for i in range(TOTAL // AUTOSAVE):
        results = Parallel(n_jobs = AGENTS)(delayed(Training)(AUTOSAVE // AGENTS, epsilon) for i in range(AGENTS))
        for state in results:
            GLOBALQ = Merge(GLOBALQ, state, AGENTS)
        print("Saving: DO NOT SHUT OFF")
        joblib.dump(dict(GLOBALQ), "q_table.pkl", compress=0)
        progress += AUTOSAVE
        print(f"Q-table auto-saved at episode {progress / TOTAL}")
        print(len(GLOBALQ))
        epsilon *= e_decay

print("Saving Q-table...")
joblib.dump(dict(GLOBALQ), "q_table.pkl", compress=3)
print("Q-table FINISHED!")

endtime = time.time()
print(endtime - starttime)