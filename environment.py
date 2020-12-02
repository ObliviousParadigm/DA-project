import pandas as pd
import datetime as dt
import numpy as np
import os
import pickle

clean_path = 'Cleaned/'

# Exponential Decay Average - EDA

class StockMarket():

    def __init__(self, symbols=['AMD', 'NFLX'], starting_cash=1000, training=True, alp=0.9):
        '''
        Initializing with stock names, the amount of money the model will start with,
        whether it is training or testing, and the value of alpha for calculating EDA
        '''
        self.symbols = symbols
        self.starting_cash = starting_cash
        self.training = training
        self.alp = alp
        self.df_companies = {}  # This dictionary will hold the dataset
        self.test_states = [0, 0]  # Stores the EDA
        for s in self.symbols:
            self.df_companies[s] = pd.read_csv(clean_path + s + '.csv')
        print('Loaded Data')
        # If the model is training, use the dataset
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

        self.iter = 1 # Calculate the EDA
        self.cur_idx = self.start_idx
        self.state = np.zeros((1, 8))
        self.portfo = 0
        # set_state has 3 arguments: Number of starting stocks of 1st stock, 2nd stock, and starting cash.
        # We are using max of 0 or the random number in order to avoid getting any negative numbers as one
        # cannot have negative stocks
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
        '''
        This function is used to calculate the portfolio
        It is done by calculating (number of stocks you have * closing price of the same stock)
        '''
        i = 0
        ret = 0
        for s in self.symbols:
            ret += self.state[0][i] * \
                self.df_companies[s][' Close/Last'].loc[self.cur_idx]
            i += 1

        ret += self.state[0][2]

        return ret

    def get_data(self):
        '''
        This function is used to get the opening prices, EDAs, and other data
        '''
        # a[0] is opening price of 1st Stock
        # a[1] is opening price of 2nd Stock
        # a[2] is EDA of a[0]
        # a[3] is EDA of a[1]
        a = self.state[0][3:7]
        i = 0
        for s in self.symbols:
            a[i] = self.df_companies[s][' Open'].loc[self.cur_idx]
            '''
            Calculate the EDA while training
            While testing, use the saved states
            '''
            if self.training:
                a[i+2] = (a[i+2] * self.alp + a[i] * (1 - self.alp)
                          ) / (1 - self.alp**self.iter)
            else:
                a[i+2] = self.test_states[i]
            i += 1
        self.iter += 1
        return a, self.df_companies[s]['Date'][self.cur_idx]

    def print_state(self):
        '''
        This function is just used to print each state
        '''
        print('Current State')
        print('-------------------------------------------------------------------------------------')
        print('| Date          | {}   | {}   | Cash      | Open_{} | Open_{}  | PortFolio |'.format(
            self.symbols[0], self.symbols[1], self.symbols[0], self.symbols[1]))

        print('| {}    |   {}  |  {}  |  {:.2f}  |  {:.2f}    |  {:.2f}    |  {:.2f} |'.format(
            self.cur_date, self.state[0][0], self.state[0][1], self.state[0][2], self.state[0][3], self.state[0][4], self.state[0][7]))

        print('-------------------------------------------------------------------------------------')

        # print(self.state[0][5], self.state[0][6])

    def step(self, action, verbose=False):
        '''
        This function is basically about the action the model wants to take.
        Buy, sell, or hold
        '''
        action = np.argmax(action[0])
        decsize = 1
        ts_left = self.cur_idx - self.end_idx
        cur_val = self.get_port()
        gain = cur_val - self.sportfolio
        
        # Return statement order in the upcoming lines
        # return new_state, reward, done flag, info(for debugging)

        if self.cur_idx <= self.end_idx:
            self.set_state(self.state[0][0],
                           self.state[0][1], self.state[0][2])
            div_bonus = 0  # Diversification bonus
            
            '''
            If the model buys atleast 1 stock from each of the companies,
            it will get a diversification bonus. This is an incentive for 
            the model so as to make sure that it does not play completely 
            safe and buy nothing nor will it risk everything and lose 
            everything
            '''
            if self.state[0][0] > 0 and self.state[0][1] > 0:
                div_bonus = 10
            new_state = self.state
            return new_state, cur_val + div_bonus + gain, True, 'DONE'

        if action == 2:
            '''
            Hold. Go to new state
            '''
            if verbose:
                print('Hold')
            self.cur_idx -= 1
            self.set_state(self.state[0][0],
                           self.state[0][1], self.state[0][2])
            new_state = self.state
            return new_state, -ts_left+gain, False, 'Hold'

        if action == 0:
            '''
            Buy stock of first company
            decsize - decision size, ie number of stocks to buy
            If the amount of stocks the model buys is more than 
            the amount of money it has, it'll go bankrupt and 
            gain is halved
            If the amount of stocks bought is valid, it's updated 
            and the amount of money left is reduced
            '''
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
            '''
            Buy stock of second company
            If the amount of stocks the model buys is more than 
            the amount of money it has, it'll go bankrupt and 
            gain is halved
            If the amount of stocks bought is valid, it's updated 
            and the amount of money left is reduced
            '''
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
            '''
            Sell stock of first company
            If the model sells too many stocks, ie more stocks than it has,
            it will be considered as an illegal action. A penalty will be 
            given.
            If the model sells an amount that is valid, it decreases the 
            number of stocks the model has of that company and increases 
            the amount of money it has
            '''
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
            '''
            Sell stock of second company
            If the model sells too many stocks, ie more stocks than it has,
            it will be considered as an illegal action. A penalty will be 
            given.
            If the model sells an amount that is valid, it decreases the 
            number of stocks the model has of that company and increases 
            the amount of money it has
            '''
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
        '''
        Reset
        '''
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
        '''
        Save the state
        '''
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
