"""
Little revision from
https://github.com/hunkim/ReinforcementZeroToAll/blob/master/07_3_dqn_2015_cartpole.py
"""

import numpy as np
import tensorflow as tf
import random
from collections import deque
import dqn

import gym
from typing import List
import matplotlib.pyplot as plt

env = gym.make('CartPole-v0')

# Constants defining our neural network
INPUT_SIZE = env.observation_space.shape[0]
OUTPUT_SIZE = env.action_space.n

DISCOUNT_RATE = 0.99
REPLAY_MEMORY = 50000
BATCH_SIZE = 64
TARGET_UPDATE_FREQUENCY = 5
MAX_EPISODES = 500


def replay_train(mainDQN, targetDQN, train_batch):
    """Trains `mainDQN` with target Q values given by `targetDQN`
    
    Args:
        mainDQN (dqn.DQN): Main DQN that will be trained()
        targetDQN (dqn.DQN): Target DQN that will predict Q_target
        train_batch (list): Minibatch of replay memory
            Each element is (s, a, r, s', done)
            [(state, action, reward, next_state, done), ...]
    
    Returns:
        float: After updating `mainDQN`, it returns a `loss`
    """
    states = np.vstack([x[0] for x in train_batch])
    actions = np.array([x[1] for x in train_batch])
    rewards = np.array([x[2] for x in train_batch])
    next_states = np.vstack([x[3] for x in train_batch])
    done = np.array([x[4] for x in train_batch])

    X = states

    # Get Q value from target network
    Q_target = rewards + DISCOUNT_RATE * np.max(targetDQN.predict(next_states), axis=1) * ~done

    y = mainDQN.predict(states)
    y[np.arange(len(X)), actions] = Q_target


    # Train our main network using target and predicted Q values on each episode
    return mainDQN.update(X, y)


def get_copy_var_ops(dest_scope_name, src_scope_name):
    """Creates TF operations that copy weights from `src_scope` to `dest_scope`
    
    Args:
        dest_scope_name (str): Destination weights (copy to)
        src_scope_name (str): Source weight (copy from)
    
    Returns:
        List[tf.Operation]: Update operations are created and returned
    """
    # Copy variables src_scope to dest_scope
    op_holder = []

    # Get 'TRAINABLE_VARIABLES' (weights) in corresponding scope
    src_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope=src_scope_name)
    dest_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope=dest_scope_name)

    # .assign(): assign value to tensor
    # .value(): get value of tensor
    for src_var, dest_var in zip(src_vars, dest_vars):
        op_holder.append(dest_var.assign(src_var.value())) # dest_var = src_val

    return op_holder


def main():
    # store the previous observations in replay memory
    replay_buffer = deque(maxlen=REPLAY_MEMORY)

    with tf.Session() as sess:
        # seperate networks
        mainDQN = dqn.DQN(sess, INPUT_SIZE, OUTPUT_SIZE, name="main")
        targetDQN = dqn.DQN(sess, INPUT_SIZE, OUTPUT_SIZE, name="target")
        sess.run(tf.global_variables_initializer())

        # initial copy q_net -> target_net
        copy_ops = get_copy_var_ops(dest_scope_name="target",
                                    src_scope_name="main")

        sess.run(copy_ops)

        step_list = []
        with open('long/log_ddqn', 'w') as f:
            for episode in range(MAX_EPISODES):
                e = 1. / ((episode / 10) + 1)
                done = False
                step_count = 0
                state = env.reset()

                while not done:
                    if np.random.rand() < e:
                        action = env.action_space.sample()
                    else:
                        # Choose an action by greedily from the Q-network
                        action = np.argmax(mainDQN.predict(state))

                    # Get new state and reward from environment
                    next_state, reward, done, _ = env.step(action)

                    if done:  # Penalty
                        reward = -1

                    # Save the experience to our buffer
                    replay_buffer.append((state, action, reward, next_state, done))

                    if len(replay_buffer) > BATCH_SIZE:
                        minibatch = random.sample(replay_buffer, BATCH_SIZE)
                        loss, _ = replay_train(mainDQN, targetDQN, minibatch)

                    if step_count % TARGET_UPDATE_FREQUENCY == 0:
                        sess.run(copy_ops)

                    state = next_state
                    step_count += 1

                f.write("Episode\t{}\tSteps\t{}\n".format(episode, step_count))
                step_list.append(step_count)

    plt.bar(range(len(step_list)), step_list, color="blue")
    plt.show()


if __name__ == "__main__":
    main()
