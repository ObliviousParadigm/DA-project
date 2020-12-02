import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras import Model
from tensorflow.keras.layers import LSTM, Dense, GRU, BatchNormalization
from tensorflow.keras.optimizers import Adam
from environment import StockMarket


class ActorCritic(Model): #after inheritance class ActorCritic becomes a Model class
    def __init__(self, inp_size): #function to initialize all the layers 
        self.inp_size = inp_size
        super(ActorCritic, self).__init__()
        self.hlayer1 = Dense(128, activation='relu') # creating objects of class Dense, first hidden layer, with 128 nodes and relu activation fucntion
        self.hlayer2 = Dense(128, activation='relu') # second hidden layer, the number 128 and activation function relu were chosen based on performance comparisons
        self.rec_layer = GRU(128, stateful=True) # gated recurrent unit, used to handle time series data and RNN, has logistic regression units as gates.
        self.hlayer3 = Dense(64, activation='relu') # third hidden layer with 64 nodes.
        self.action_layer = Dense(5, activation='softmax') # predicts what actions to take, 5 is the no of actions, indexed from 0 to 4
        #0 buy first stock; 1: sell first stock; 2: hold; 3:buy second; 4: sell second; 
        self.critic_layer = Dense(1) # tries to find the action that gives best results, is a real number so no activation function is used.

    def call(self, inputs): # part of the class ActorCritic which executes whenever model() is called
        x = tf.convert_to_tensor(inputs) # the inputs has to be converted to tensors to use tensor flow keras.
        x = self.hlayer1(x) # callable objects, not a function call, keras allows the use of callable objects
        x = self.hlayer2(x) # output of first layer is input to the second.
        x = self.rec_layer(x) # output of second layer is input to the GRU layer and so on till layer 3
        x = self.hlayer3(x) #so far there was no branching, next line is start of a new branch
        actions = self.action_layer(x) 
        statev = self.critic_layer(x) # stores the real number value of the state.
        #debugging and error handling, NANs, infinity etc are taken for computation by tensorflow and results in NANs in the output or wrong ouput
        try:
            tf.debugging.check_numerics(actions, 'Output') #check the actions to detect NANs
        except:
            print(x) #the value of x gives an idea of possible bug. 
            raise IndentationError #raising some error to stop execution 
        return actions, statev 


def epsilon_greedy(a, eps):
    p = np.random.random()
    if p >= eps:
        return a
    else:
        ra = np.zeros((1, 5))
        i = np.random.choice([0, 1, 2, 3, 4])
        ra[0][i] = 1
        return ra


def train(n_episodes=100, eps=0.8, gamma=0.9, r_decay=0.2):
    i_eps = eps
    env = StockMarket()
    model = ActorCritic(8)
    opt = Adam()
    running_reward = 0
    s = env.reset()
    s = np.reshape(s, (1, 1, s.shape[1]))
    a = model(s)
    model._set_inputs(s)
    ckpt = tf.train.Checkpoint(optimizer=opt, model=model)
    manager = tf.train.CheckpointManager(
        ckpt, directory="mdl_saved/", max_to_keep=10)
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")
    for i in range(n_episodes):
        running_loss = 0
        eps = i_eps * (0.9**i)
        print("Episode {} start".format(i+1))
        s = env.reset()
        s = np.reshape(s, (1, 1, s.shape[1]))
        r = 0
        done = False
        info = ''
        j = 0
        while not done:
            # if j % 100 == 0:
            #     env.print_state()
            with tf.GradientTape() as tape:
                a, v = model(s)
                new_s, r, done, info = env.step(epsilon_greedy(a, eps))
                new_s = np.reshape(new_s, (1, 1, new_s.shape[1]))
                _, new_v = model(new_s)
                if done:
                    delta = (r - v)
                else:
                    delta = (r + gamma*tf.stop_gradient(new_v) - v)
                critic_loss = delta**2
                actor_loss = -(tf.stop_gradient(delta)*tf.math.log(a+1e-7))
                loss = actor_loss + critic_loss
                loss = tf.clip_by_value(loss, 1e-5, 1e5)
            grads = tape.gradient(loss, model.trainable_variables)
            running_loss += np.sum(loss)
            opt.apply_gradients(zip(grads, model.trainable_variables))
            running_reward = r_decay*running_reward + (1-r_decay)*r
            s = new_s
            j += 1
        #print("Action for debugging: ", a)
        print("Info: ", info)
        print("Episode {} compeleted, running reward {}".format(i+1, running_reward))
        print("Loss of model: ", running_loss/j)
        model.rec_layer.reset_states()
        if i % 100 == 0:
            manager.save(i)
        # if i % 100 == 0:
        #     model.save_weights(checkpoint_path.format(episode=i))
    model.save_weights('mdl_final/')
    env.save()
    print("Training is finished Model is saved")


if __name__ == '__main__':
    train(10000)
