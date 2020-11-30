import tensorflow as tf
from environment import StockMarket
from training_model import ActorCritic
import numpy as np
import matplotlib.pyplot as plt

logdir = 'logs/'


if __name__ == '__main__':
    # writer = tf.summary.create_file_writer(logdir)
    env = StockMarket(training=False)
    s = env.reset()
    s = np.reshape(s, (1, 1, s.shape[1]))
    model = ActorCritic(8)
    a = model(s)
    model._set_inputs(s)
    ckpt = tf.train.Checkpoint(model=model)
    ckpt.restore(tf.train.latest_checkpoint(
        'mdl_saved/ckpt-400')).expect_partial()

    # tf.summary.trace_on(graph=True)
    # model(s)
    # with writer.as_default():
    #     tf.summary.trace_export(name='model', step=0)
    # model.rec_layer.reset_states()
    print("Test Start")
    g = []
    for i in range(50):
        s = env.reset()
        s = np.reshape(s, (1, 1, s.shape[1]))
        r = 0
        done = False
        while not done:
            env.print_state()
            a, _ = model(s)
            new_s, r, done, _ = env.step(a, verbose=True)
            new_s = np.reshape(new_s, (1, 1, new_s.shape[1]))
            s = new_s
        model.rec_layer.reset_states()
        g.append(env.get_port() - env.sportfolio)
        print("Gain: ", g[-1], " in episode ", i+1)

    plt.boxplot(g)
    plt.ylabel('Prices')
    plt.show()
    print("Average Gain: ", np.mean(g))
    print("Testing Finished!!")
