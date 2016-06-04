__author__ = 'trevorlindsay'

import json
import numpy as np
from collections import deque, defaultdict


class Bot(object):

    def __init__(self):

        self._iteration = 0
        self._last_state = None
        self._last_action = None
        self._moves = deque([])
        self._reward = (-1, 100, -1000)
        self._qvalues = defaultdict(self.init_qvalues)
        self.load_qvalues()
        self._last_score = 0
        self._training_data = []
        self.score = 0
        self.discount_rate = 0.5
        self.learning_rate = 0.95


    def load_qvalues(self):
        try: self.qvalues.update(json.load(open('qvalues.json', 'rb')))
        except: return


    @staticmethod
    def init_qvalues():
        return [0,0]


    def dump_qvalues(self):
        json.dump(self.qvalues, open('qvalues.json', 'wb'))


    def reset(self):
        self._moves = deque([])
        self._training_data.append((self.iteration, self.score))
        self.score = 0
        self._iteration += 1


    def get_qvalue(self, player_x, player_y, upperpipe_x, upperpipe_y, vel):

        round_to = 10 if player_x - upperpipe_x > -300 else 50
        x_dist = int(player_x) - int(upperpipe_x % round_to)
        y_dist = int(np.round(player_y - upperpipe_y, -1))
        state = '{}:{}:{}'.format(x_dist, y_dist, vel)

        return state, self.qvalues[state]


    def get_action(self, player_x, player_y, upperpipe_x, upperpipe_y, vel):

        state, qvalue = self.get_qvalue(player_x, player_y, upperpipe_x, upperpipe_y, vel)
        action = np.argmax(qvalue) if qvalue[0] != qvalue[-1] else 0
        score = 0 if self.score == self.last_score else 1

        if self.last_state and self._last_action:
            self.update_qvalue(score, state)

        if score:
            for i, (state, action) in enumerate(self.moves):
                # Backpropagate the reward
                reward = 1000
                # self.qvalues[state][action] += reward ** (1. / (i + 1))

        self._last_state = state
        self._last_action = action
        self._last_score = self.score

        self._moves.appendleft((state, action))

        return action


    def update_qvalue(self, score, state):

        if state:
            self._qvalues[self.last_state][self.last_action] += \
                self.learning_rate * (self.reward[score] + self.discount_rate * np.max(self.qvalues[state]) - self._qvalues[self.last_state][self.last_action])
        else:
            self._qvalues[self.last_state][self.last_action] += \
                self.learning_rate * (self.reward[score] - self._qvalues[self.last_state][self.last_action])


    def onCrash(self, ground=False, player_y=0, pipe1=0, pipe2=0, pipeH=0):

        self.update_qvalue(score=-1, state=None)
        self.dump_qvalues()

        if ground or np.abs(player_y - (pipe1 + pipeH + pipe2) / 2) > 100:
            for i, (state, action) in enumerate(self.moves):
                # Backpropagate the penalty
                self._qvalues[state][action] -= np.abs(self.reward[-1]) ** (self.learning_rate * (1. / (i + 1)))

        print 'Iteration: {}, Score: {}'.format(self.iteration, self.score)


    def dump_training_data(self):
        json.dump(self.training_data, open('training_data.json', 'wb'))


    @property
    def qvalues(self):
        return self._qvalues

    @property
    def iteration(self):
        return self._iteration

    @property
    def last_state(self):
        return self._last_state

    @property
    def last_action(self):
        return self._last_action

    @property
    def moves(self):
        return self._moves

    @property
    def reward(self):
        return self._reward

    @property
    def last_score(self):
        return self._last_score

    @property
    def training_data(self):
        return self._training_data