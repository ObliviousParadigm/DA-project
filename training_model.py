import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras import Model
from tensorflow.keras.layers import LSTM, Dense, GRU, BatchNormalization
from tensorflow.keras.optimizers import Adam
from environment import StockMarket


class ActorCritic(Model):
    def __init__(self, inp_size):
        self.inp_size = inp_size
        super(ActorCritic, self).__init__()
        self.hlayer1 = Dense(128, activation='relu')
        self.hlayer2 = Dense(128, activation='relu')
        self.rec_layer = GRU(128, stateful=True)
        self.hlayer3 = Dense(64, activation='relu')
        self.action_layer = Dense(5, activation='softmax')
        self.critic_layer = Dense(1)

    def call(self, inputs):
        x = tf.convert_to_tensor(inputs)
        x = self.hlayer1(x)
        x = self.hlayer2(x)
        x = self.rec_layer(x)
        x = self.hlayer3(x)
        actions = self.action_layer(x)
        statev = self.critic_layer(x)
        try:
            tf.debugging.check_numerics(actions, 'Output')
        except:
            print(x)
            raise IndentationError
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
