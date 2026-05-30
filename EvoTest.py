import random

import joblib
import numpy as np
import os
from collections import defaultdict
from collections import OrderedDict
from copy import deepcopy
from joblib import Parallel, delayed
import bisect

def DefaultQ():
    return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

GLOBALQ = defaultdict(DefaultQ)
if os.path.exists("q_table.pkl"):
    with open("q_table.pkl", "rb") as f:
        GLOBALQ = deepcopy(defaultdict(DefaultQ, joblib.load("q_table.pkl")))

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
        self.updated = {}

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
    def __init__(self, maxsize=500000):
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
    q = env.Q[state].copy()
    for i in range(13):
        if not i + 1 in hand:
            q[i] = -9999
    if np.random.rand() < epsilon and len(hand):
        return np.random.choice(hand)
    else:
        return np.argmax(q) + 1

def Training(TRIALS, epsilon):
    env = GoFishEnv()
    LRU = LRU_CACHE()
    wins = 0
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

                won = env.Question(action, env.Hand1, env.Hand2)
                if action in env.Hand1:
                    env.potential2[action] += 1
                    env.potential1[action] = 0
                else:
                    env.Q[state][action - 1] -= 0.6
                if won:
                    points = env.CheckBooks(env.Hand1)
                    env.Score1 += points

                if not won:
                    if len(env.Cards) > 0:
                        env.Draw(env.Hand1)
                        while len(env.Hand1) == 0:
                            env.Draw(env.Hand1)
                        env.Score1 += env.CheckBooks(env.Hand1)
                    P1 = False
            while P2:
                if env.Score1 + env.Score2 == 13:
                    break
                if len(env.Hand2) == 0:
                    env.Draw(env.Hand2)
                if len(env.Hand1) == 0:
                    env.Draw(env.Hand1)
                action = random.choice(env.Hand2)

                won = env.Question(action, env.Hand2, env.Hand1)
                if action in env.Hand2:
                    env.potential1[action] += 1
                    env.potential2[action] = 0
                if won:
                    points = env.CheckBooks(env.Hand2)
                    env.Score2 += points

                if not won:
                    if len(env.Cards) > 0:
                        env.Draw(env.Hand2)
                        while len(env.Hand2) == 0:
                            env.Draw(env.Hand2)
                        env.Score2 += env.CheckBooks(env.Hand2)
                    P2 = False
        if env.Score1 > env.Score2:
            wins += 1
    return wins

TOTAL = 100000
print(Training(TOTAL, 0)/TOTAL)
