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
        self._reward = (1, -100, -1000)
        self._qvalues = defaultdict(self.init_qvalues)
        self.load_qvalues()
        self._last_score = 0
        self._training_data = []
        self.action_count = 0
        self.score = 0
        self.discount_rate = 1.0
        self.lr = 0.95


    def load_qvalues(self):
        try: self.qvalues.update(json.load(open('qvalues1.json', 'rb')))
        except: return


    @staticmethod
    def init_qvalues():
        return [0,0]


    def dump_qvalues(self):
        json.dump(self.qvalues, open('qvalues1.json', 'wb'))


    def reset(self):
        self._moves = deque([])
        self._training_data.append((self.iteration, self.score))
        self.score = 0
        self.action_count = 0
        self._iteration += 1


    def get_qvalue(self, player_x, player_y, upperpipe_x, upperpipe_y, vel):

        round_to = 10 if player_x - upperpipe_x > -300 else 10
        x_dist = int(player_x - upperpipe_x)
        x_dist = int(x_dist - (x_dist % round_to))
        y_dist = int(np.round(player_y - upperpipe_y, -1))
        state = '{}:{}:{}'.format(x_dist, y_dist, vel)

        return state, self.qvalues[state]


    def get_action(self, player_x, player_y, upperpipe_x, upperpipe_y, vel):

        state, qvalue = self.get_qvalue(player_x, player_y, upperpipe_x, upperpipe_y, vel)
        action = np.argmax(qvalue) if qvalue[0] != qvalue[-1] else 0
        score = 0 if self.score == self.last_score else 1

        if self.last_state:
            self._moves.appendleft((self.last_state, self.last_action, state))

        self._last_state = state
        self._last_action = action
        self._last_score = self.score
        self.action_count += 1

        return action


    def update_qvalue(self, last_state, action, new_state, score):
        self._qvalues[last_state][action] +=  \
                self.lr * (self.reward[score] + self.discount_rate * max(self.qvalues[new_state]))


    def onCrash(self, ground=False, playerY=0, pipeY=0, pipeH=0):

        for i, (state, action, new_state) in enumerate(self.moves):
            if i <= 2:
                self.update_qvalue(state, action, new_state, 2)
            elif ground or (action == 1 and playerY < (pipeY + pipeH)):
                self.update_qvalue(state, action, new_state, 1)
            else:
                self.update_qvalue(state, action, new_state, 0)

        print 'Iteration: {}, Score: {}, Actions: {}'.format(self.iteration, self.score, self.action_count)

        if self.iteration % 10 == 0:
            self.dump_qvalues()
            self.dump_training_data()


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