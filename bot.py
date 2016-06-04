__author__ = 'trevorlindsay'

import json
import numpy as np
from collections import deque


class Bot(object):

    def __init__(self, pipegapsize):

        self._iteration = 0
        self._last_state = None
        self._last_action = None
        self._current_state = None
        self._actions = deque([])
        self._states = deque([])
        self._score = 0
        self._reward = (0, 100)
        self._qvalues = self.load_qvalues()
        self._pipegapsize = pipegapsize
        self.discount_rate = 1.0

    def reset(self, score=True):
        self._states, self._actions  = deque([]), deque([])
        if score:
            self._score = 0

    def get_action(self, player_x, player_y, upperpipe_x, upperpipe_y, vel):

        state, qvalue = self.get_qvalue(player_x, player_y, upperpipe_x, upperpipe_y, vel)
        action = np.argmax(qvalue) if qvalue[0] != qvalue[-1] else 0
        score = self.get_score(player_x, upperpipe_x)

        if self.last_state and self._last_action:
            self.update_qvalue(score, state)

        if score:

            for i, (state, action) in enumerate(zip(self.states, self.actions)):
                qvalue = self.qvalues.get(state, [0, 0])
                qvalue[action] += 100 ** (1 / (i + 1))
                self._qvalues[state] = qvalue

            self.reset(score=False)

        self._last_state = state
        self._last_action = action
        self._score += score

        self._states.appendleft(state)
        self._actions.appendleft(action)

        return action

    def update_crash(self):

        qvalue = self.qvalues.get(self.last_state, [0,0])
        qvalue[self.last_action] = -1000
        self._qvalues[self.last_state] = qvalue

        self.dump_qvalues()
        self._iteration += 1

        for i, (state, action) in enumerate(zip(self.states, self.actions)):
            qvalue = self.qvalues.get(state, [0,0])
            qvalue[action] += -100 ** (1/(i + 1))
            self._qvalues[state] = qvalue

        print 'Iteration: {}, Score: {}'.format(self.iteration, self._score)

    def update_qvalue(self, score, state):
        self._qvalues[self.last_state][self.last_action] = \
            self.reward[score] + self.discount_rate * np.max(self.qvalues.get(state, [0,0]))

    @staticmethod
    def get_score(player_x, upperpipe_x):

        if upperpipe_x <= player_x < upperpipe_x + 4:
            return 1
        else:
            return 0

    def get_qvalue(self, player_x, player_y, upperpipe_x, upperpipe_y, vel):

        x_dist = int(np.round(player_x - upperpipe_x, -1))
        y_dist = int(np.round(player_y - upperpipe_y, -1))
        state = '{}:{}:{}'.format(x_dist, y_dist, vel)

        return state, self.qvalues.get(state, [0,0])

    def load_qvalues(self):

        try:
            return json.load(open('qvalues.json', 'rb'))
        except:
            return self.init_qvalues()

    def dump_qvalues(self):
        json.dump(self.qvalues, open('qvalues.json', 'wb'))

    @staticmethod
    def init_qvalues():

        qvalues = {}
        for x in range(-430, 40, 10):
            for y in range(300, 560, 10):
                for v in range(-9, 11, 1):
                    qvalues['{}:{}:{}'.format(x, y, v)] = [0, 0]

        return qvalues

    @property
    def pipegapsize(self):
        return self._pipegapsize

    @property
    def qvalues(self):
        return self._qvalues

    @property
    def iteration(self):
        return self._iteration

    @property
    def current_state(self):
        return self._current_state

    @property
    def last_state(self):
        return self._last_state

    @property
    def last_action(self):
        return self._last_action

    @property
    def actions(self):
        return self._actions

    @property
    def reward(self):
        return self._reward

    @property
    def states(self):
        return self._states