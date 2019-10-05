"""
PER based on
https://github.com/hunkim/ReinforcementZeroToAll/blob/master/07_3_dqn_2015_cartpole.py
"""

import numpy as np
import tensorflow as tf
import random
from collections import deque
import per
import sumtree

import gym
from typing import List
import matplotlib.pyplot as plt

env = gym.make('CartPole-v0')

# Constants defining our NN
INPUT_SIZE = env.observation_space.shape[0]
OUTPUT_SIZE = env.action_space.n

DISCOUNT_RATE = 0.99
REPLAY_MEMORY = 50000
BATCH_SIZE = 64
TARGET_UPDATE_FREQUENCY = 5
MAX_EPISODES = 500

EPSILON = 0.00001
ALPHA = 0.6
BETA = 0.4


def replay_train(mainDQN, targetDQN, minibatch, replay_buffer):
    states = np.vstack([x[1][0] for x in minibatch])
    actions = np.array([x[1][1] for x in minibatch])
    rewards = np.array([x[1][2] for x in minibatch])
    next_states = np.vstack([x[1][3] for x in minibatch])
    done = np.array([x[1][4] for x in minibatch])
    prio = np.array([x[1][5] for x in minibatch])

    X = states

    Q_target = rewards + \
        DISCOUNT_RATE * np.max(targetDQN.predict(next_states), axis=1) * \
         ~done
    Y = mainDQN.predict(states)
    Y[np.arange(len(X)), actions] = Q_target

    # Set ISW (Important Sampling Weight)
    priosum = 0
    for exp in replay_buffer:
        priosum += exp[5]
    prob = prio / priosum

    global BETA
    isw = np.power(BATCH_SIZE * prob, BETA)
    isw = 1 / isw

    loss, train = mainDQN.update(X, Y, isw)

    # Update TD_error
    Y2 = mainDQN.predict(states)[0][actions]

    new_prio = abs(Q_target - Y2) + EPSILON # |TDerror| + EPSIOLON
    new_prio = pow(new_prio, ALPHA)

    i = 0
    for x in minibatch:
        replay_buffer[x[0]][5] = new_prio[i]
        i += 1

    '''
    for x in minibatch:
        temp_exp = replay_buffer[x[0]]

        q_target = (temp_exp[2] + \
            DISCOUNT_RATE * np.max(targetDQN.predict(temp_exp[3]), axis=1) * \
            ~ temp_exp[4])[0]
        q_pred = mainDQN.predict(temp_exp[0])[0][temp_exp[1]]

        prio = abs(q_target - q_pred) + EPSILON # |TDerror| + EPSIOLON
        prio = pow(prio, ALPHA)

        replay_buffer[x[0]][5] = prio
    '''

    # Anneal up BETA to 1
    BETA = BETA + (1 - BETA) / MAX_EPISODES

    return replay_buffer, loss, train


def get_copy_var_ops(dest_scope_name, src_scope_name):
    # Copy variables src_scope to dest_scope
    op_holder = []

    # Get 'TRAINABLE_VARIABLES' (= weights) in corresponding scope
    src_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope=src_scope_name)
    dest_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope=dest_scope_name)

    # .assign(): assign value to tensor
    # .value(): get value of tensor
    for src_var, dest_var in zip(src_vars, dest_vars):
        op_holder.append(dest_var.assign(src_var.value()))

    return op_holder


def get_prioritized_minibatch(replay_buffer):
    # Make sumtree
    my_tree = sumtree.SumTree(len(replay_buffer))

    # Complete sumtree
    for i in range(len(replay_buffer)):
        my_tree.add(replay_buffer[i][5])

    # Make minibatch
    my_tree_total = my_tree.total()
    minibatch = []

    for i in range(BATCH_SIZE):
        s = random.uniform(0, my_tree_total)
        dataIdx = my_tree.get(s)
        minibatch.append([dataIdx, replay_buffer[dataIdx]])

    return minibatch


def main():
    # Replay memory to store previous observations
    replay_buffer = deque(maxlen=REPLAY_MEMORY)

    with tf.Session() as sess:
        # Separate networks
        mainDQN = per.DQN(sess, INPUT_SIZE, OUTPUT_SIZE, name="main")
        targetDQN = per.DQN(sess, INPUT_SIZE, OUTPUT_SIZE, name="target")
        sess.run(tf.global_variables_initializer())

        # Initial copy q_net to target_net
        # Copying network means copying weights in NN
        copy_ops = get_copy_var_ops(dest_scope_name="target", src_scope_name="main")
        sess.run(copy_ops)

        step_list = []
        with open('log/log_per', 'w') as f:
            for episode in range(MAX_EPISODES):
                e = 1. / ((episode / 10) + 1)
                done = False
                step_count = 0
                state = env.reset()

                while not done:
                    # Choose action with 'Explore and Exploit'
                    if np.random.rand() < e:
                        action = env.action_space.sample()
                    else:
                        action = np.argmax(mainDQN.predict(state))

                    next_state, reward, done, _ = env.step(action)

                    if done: # Penalty
                        reward = -1

                    # Calculate (manitude of) TD error
                    q_target = (reward + \
                        DISCOUNT_RATE * np.max(targetDQN.predict(next_state), axis=1) * \
                        ~ done)[0]
                    q_pred = mainDQN.predict(state)[0][action]
                    prio = abs(q_target - q_pred) + EPSILON # |TDerror| + EPSIOLON
                    prio = pow(prio, ALPHA)

                    # Save the experience to our buffer
                    replay_buffer.append([state, action, reward, next_state, done, prio])

                    # Experience replay
                    if len(replay_buffer) > BATCH_SIZE:
                        minibatch = get_prioritized_minibatch(replay_buffer)
                        replay_buffer, loss, _ = replay_train(mainDQN, targetDQN, 
                            minibatch, replay_buffer)

                    # Update target network
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
