import pandas as pd
import datetime as dt
import numpy as np
import os
import pickle

clean_path = 'Cleaned/'


class StockMarket():

    def __init__(self, symbols=['AMD', 'NFLX'], starting_cash=1000, training=True, alp=0.9):
        self.symbols = symbols
        self.starting_cash = starting_cash
        self.training = training
        self.alp = alp
        self.df_companies = {}
        self.test_states = [0, 0]
        for s in self.symbols:
            self.df_companies[s] = pd.read_csv(clean_path + s + '.csv')
        print('Loaded Data')
        if self.training:
            self.start_idx = len(self.df_companies[self.symbols[0]]) - 1
            self.end_idx = round(len(self.df_companies) * 0.2)
        else:
            with open('states.txt', 'rb') as f:
                l = pickle.load(f)

            self.test_states = l
            self.start_idx = round(
                len(self.df_companies[self.symbols[0]]) * 0.2) - 1
            self.end_idx = 1

        self.iter = 1
        self.cur_idx = self.start_idx
        self.state = np.zeros((1, 8))
        self.portfo = 0
        self.set_state(max(int(np.random.normal(100)), 0), max(
            int(np.random.normal(100)), 0), self.starting_cash)
        self.sportfolio = self.state[0][7]

    def set_state(self, s1, s2, cash):
        self.state[0][0] = s1
        self.state[0][1] = s2
        self.state[0][2] = cash
        self.state[0][3:7], self.cur_date = self.get_data()
        self.state[0][7] = self.get_port()

    def get_port(self):
        i = 0
        ret = 0
        for s in self.symbols:
            ret += self.state[0][i] * \
                self.df_companies[s][' Close/Last'].loc[self.cur_idx]
            i += 1

        ret += self.state[0][2]

        return ret

    def get_data(self):
        a = self.state[0][3:7]
        i = 0
        for s in self.symbols:
            a[i] = self.df_companies[s][' Open'].loc[self.cur_idx]
            if self.training:
                a[i+2] = (a[i+2] * self.alp + a[i] * (1 - self.alp)
                          ) / (1 - self.alp**self.iter)
            else:
                a[i+2] = self.test_states[i]
            i += 1
        self.iter += 1
        return a, self.df_companies[s]['Date'][self.cur_idx]

    def print_state(self):
        print('Current State')
        print('-------------------------------------------------------------------------------------')
        print('| Date          | {}   | {}   | Cash      | Open_{} | Open_{}  | PortFolio |'.format(
            self.symbols[0], self.symbols[1], self.symbols[0], self.symbols[1]))

        print('| {}    |   {}  |  {}  |  {:.2f}  |  {:.2f}    |  {:.2f}    |  {:.2f} |'.format(
            self.cur_date, self.state[0][0], self.state[0][1], self.state[0][2], self.state[0][3], self.state[0][4], self.state[0][7]))

        print('-------------------------------------------------------------------------------------')

        # print(self.state[0][5], self.state[0][6])

    def step(self, action, verbose=False):
        action = np.argmax(action[0])
        decsize = 1
        ts_left = self.cur_idx - self.end_idx
        cur_val = self.get_port()
        gain = cur_val - self.sportfolio

        if self.cur_idx <= self.end_idx:
            self.set_state(self.state[0][0],
                           self.state[0][1], self.state[0][2])
            div_bonus = 0
            if self.state[0][0] > 0 and self.state[0][1] > 0:
                div_bonus = 10
            new_state = self.state
            return new_state, cur_val + div_bonus + gain, True, 'DONE'

        if action == 2:
            if verbose:
                print('Hold')
            self.cur_idx -= 1
            self.set_state(self.state[0][0],
                           self.state[0][1], self.state[0][2])
            new_state = self.state
            return new_state, -ts_left+gain, False, 'Hold'

        if action == 0:
            if verbose:
                print('Buy {} of {}'.format(decsize, self.symbols[0]))
            if decsize * self.state[0][3] > self.state[0][2]:
                self.cur_idx -= 1
                self.set_state(self.state[0][0],
                               self.state[0][1], self.state[0][2])
                new_state = self.state
                return new_state, -ts_left+gain/2, True, 'Bankrupted'
            else:
                s1 = self.state[0][0] + decsize
                cash_spent = decsize * self.state[0][3]
                self.cur_idx -= 1
                self.set_state(s1,
                               self.state[0][1], self.state[0][2] - cash_spent)
                new_state = self.state
                return new_state, -ts_left+gain, False, 'Bought ' + self.symbols[0]
        if action == 3:
            if verbose:
                print('Buy {} of {}'.format(decsize, self.symbols[1]))
            if decsize * self.state[0][4] > self.state[0][2]:
                self.cur_idx -= 1
                self.set_state(self.state[0][0],
                               self.state[0][1], self.state[0][2])
                new_state = self.state
                return new_state, -ts_left+gain/2, True, 'Bankrupted'
            else:
                s2 = self.state[0][1] + decsize
                cash_spent = decsize * self.state[0][4]
                self.cur_idx -= 1
                self.set_state(self.state[0][0],
                               s2, self.state[0][2] - cash_spent)
                new_state = self.state
                return new_state, -ts_left+gain, False, 'Bought ' + self.symbols[1]

        if action == 1:
            if verbose:
                print('Sell {} of {}'.format(decsize, self.symbols[0]))
            if decsize > self.state[0][0]:
                self.cur_idx -= 1
                self.set_state(self.state[0][0],
                               self.state[0][1], self.state[0][2])
                new_state = self.state
                return new_state, -ts_left+gain/2, True, 'Sold Too Much'
            else:
                s1 = self.state[0][0] - decsize
                cash_gained = decsize * self.state[0][3]
                self.cur_idx -= 1
                self.set_state(s1,
                               self.state[0][1], self.state[0][2] + cash_gained)
                new_state = self.state
                return new_state, -ts_left+gain, False, 'Sold ' + self.symbols[0]

        if action == 4:
            if verbose:
                print('Sell {} of {}'.format(decsize, self.symbols[1]))
            if decsize > self.state[0][1]:
                self.cur_idx -= 1
                self.set_state(self.state[0][0],
                               self.state[0][1], self.state[0][2])
                new_state = self.state
                return new_state, -ts_left+gain/2, True, 'Sold Too Much'
            else:
                s2 = self.state[0][1] - decsize
                cash_gained = decsize * self.state[0][4]
                self.cur_idx -= 1
                self.set_state(self.state[0][0],
                               s2, self.state[0][2] + cash_gained)
                new_state = self.state
                return new_state, -ts_left+gain, False, 'Sold ' + self.symbols[1]

    def reset(self):
        self.iter = 1
        self.cur_idx = self.start_idx
        self.state = np.zeros((1, 8))
        self.portfo = 0
        self.set_state(max(int(np.random.normal(100)), 0), max(
            int(np.random.normal(100)), 0), self.starting_cash)
        self.sportfolio = self.state[0][7]
        new_state = self.state
        return new_state

    def save(self):
        with open('states.txt', 'wb') as f:
            s = list(self.state[0][5:7])
            print("Saving States: ", s)
            pickle.dump(s, f)


if __name__ == '__main__':
    env = StockMarket()
    env.print_state()
    env.step([[1, 0, 0, 0, 0]])
    env.print_state()
    env.reset()
    env.print_state()
