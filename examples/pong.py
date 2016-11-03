# coding: utf-8



# author: marco valenzuela


from __future__ import division

import gym
import numpy as np
import random
from scipy.misc import imresize
from keras.models import Model as KerasModel
from keras.layers import Input, Convolution2D, Dense, Flatten
from keras.optimizers import RMSprop
from keras import backend as K


# http://www.nature.com/nature/journal/v518/n7540/pdf/nature14236.pdf


STATE_SHAPE = (84, 84, 4)


class ReplayMemory(object):

    def __init__(self, capacity):
        # TODO maybe this would be more efficient with numpy arrays
        self.size = 0
        self.capacity = capacity
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.terminals = []

    def add(self, s_curr, a_curr, r_curr, s_next, terminal):
        if self.size > self.capacity:
            # delete first element in memory
            self.states.pop(0)
            self.actions.pop(0)
            self.rewards.pop(0)
            self.next_states.pop(0)
            self.terminals.pop(0)
            self.size -= 1
        self.states.append(s_curr)
        self.actions.append(a_curr)
        self.rewards.append(r_curr)
        self.next_states.append(s_next)
        self.terminals.append(terminal)
        self.size += 1

    def sample(self, n):
        indices = random.sample(xrange(self.size), n)
        states = []
        actions = []
        rewards = []
        next_states = []
        terminals = []
        for i in indices:
            states.append(self.states[i])
            actions.append([self.actions[i]])
            rewards.append([self.rewards[i]])
            next_states.append(self.next_states[i])
            terminals.append([self.terminals[i]])
        return (states, actions, rewards, next_states, terminals)


class Environment(object):

    def __init__(self, id):
        self.gym = gym.make(id)
        self.frames = []
        self.done = None

    def n_actions(self):
        return self.gym.action_space.n

    def random_action(self):
        return self.gym.action_space.sample()

    def render(self):
        self.gym.render()

    def is_done(self):
        return 1 if self.done else 0

    def step(self, action):
        observation, reward, done, info = self.gym.step(action)
        self.done = done
        frame = imresize(observation.sum(axis=2) / 3, (84, 84))
        self.frames.append(frame)
        if len(self.frames) == 5:
            self.frames.pop(0)
        return reward

    def reset(self):
        self.observation = self.gym.reset()
        self.done = False
        self.frames = []

    def has_state(self):
        return len(self.frames) == 4

    def get_state(self):
        state = np.empty(STATE_SHAPE)
        for i in xrange(4):
            state[:, :, i] = self.frames[i]
        return state




class Model(object):

    def __init__(self, n_actions, discount_factor, learning_rate):
        self.n_actions = n_actions
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate

        # The input to the neural network consists of an 84 × 84 × 4 image produced by the preprocessing map φ.
        input = Input(shape=STATE_SHAPE)
        # The first hidden layer convolves 32 filters of 8 × 8 with stride 4 with the input image
        # and applies a rectifier nonlinearity.
        h = Convolution2D(32, 8, 8, subsample=(4, 4), activation='relu', border_mode='same')(input)
        # The second hidden layer convolves 64 filters of 4 × 4 with stride 2, again followed
        # by a rectifier nonlinearity.
        h = Convolution2D(64, 4, 4, subsample=(2, 2), activation='relu', border_mode='same')(h)
        # This is followed by a third convolutional layer that convolves 64 filters of 3 × 3 with stride 1
        # followed by a rectifier.
        h = Convolution2D(64, 3, 3, subsample=(1, 1), activation='relu', border_mode='same')(h)
        # The final hidden layer is fully-connected and consists of 256 rectifier units.
        h = Flatten()(h)
        h = Dense(256, activation='relu')(h)
        # The output layer is a fully-connected linear layer with a single output for each valid action.
        output = Dense(n_actions)(h)

        # init model and target model
        self.model = KerasModel(input, output)
        self.copy_model_to_target()

        # declare input variables
        state = Input(shape=STATE_SHAPE)
        # action needs to be an int because it is used as an index to get the action's q-value
        action = Input(shape=(1,), dtype='int32')
        reward = Input(shape=(1,))
        next_state = Input(shape=STATE_SHAPE)
        terminal = Input(shape=(1,))

        # get q-values for states
        q_values = self.model(state)

        # action's q-value
        value = q_values[:, action]

        # get q-values for next state (after performing action in original state)
        next_q_values = self.target(next_state)

        # max q-value for next state
        max_next_q_value = K.max(next_q_values, axis=1, keepdims=True)

        # target q-value
        target = reward + (1 - terminal) * discount_factor * max_next_q_value

        # loss function: squared error, clipped to be between -1 and 1
        loss = K.mean(K.clip(K.pow(target - value, 2), -1, 1))

        # optimize with RMSProp
        optimizer = RMSprop(learning_rate)
        params = self.model.trainable_weights
        constraints = []
        updates = optimizer.get_updates(params, constraints, loss)

        # declare function to train with mini-batch
        self.train = K.function([state, action, reward, next_state, terminal], loss, updates=updates)
        self.predict = K.function([state], self.model(state))

    def copy_model_to_target(self):
        input = Input(shape=(84, 84, 4))
        h = Convolution2D(32, 8, 8, subsample=(4, 4), activation='relu', border_mode='same', weights=get_weights(self.model.layers[1]))(input)
        h = Convolution2D(64, 4, 4, subsample=(2, 2), activation='relu', border_mode='same', weights=get_weights(self.model.layers[2]))(h)
        h = Convolution2D(64, 3, 3, subsample=(1, 1), activation='relu', border_mode='same', weights=get_weights(self.model.layers[3]))(h)
        h = Flatten()(h)
        h = Dense(256, activation='relu', weights=self.model.layers[5].get_weights())(h)
        output = Dense(self.n_actions, weights=self.model.layers[6].get_weights())(h)
        self.target = KerasModel(input, output)

    def choose_action(self, state):
        return K.argmax(self.predict(state))


def get_weights(layer):
    # NOTE do we really need to copy the weights? this needs testing
    return [a.copy() for a in layer.get_weights()]



class Agent(object):

    def __init__(self, model, env, mem, initial_exploration, final_exploration, final_exploration_frame, action_repeat):
        self.model = model
        self.env = env
        self.mem = mem
        self.epsilon = initial_exploration
        self.epsilon_update = (final_exploration - initial_exploration) / final_exploration_frame
        self.final_exploration_frame = final_exploration_frame
        self.frame = 0 # TODO rename
        self.action_repeat = action_repeat

    def n_actions(self):
        return self.env.n_actions()

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return self.env.random_action()
        else:
            return self.model.choose_action(state)

    def act(self):
        state = self.env.get_state()
        action = self.choose_action(state)
        self.perform_action(action)
        if self.frame < self.final_exploration_frame:
            self.epsilon += self.epsilon_update
        self.frame += 1

    def perform_action(self, action):
        for i in xrange(self.action_repeat):
            if self.env.has_state():
                state = self.env.get_state()
                reward = self.env.step(action)
                next_state = self.env.get_state()
                terminal = self.env.is_done()
                self.mem.add(state, action, reward, next_state, terminal)
                if terminal:
                    return
            else:
                self.env.step(action)

    # initialize the memory with some random stuff
    def random_actions(self, n):
        for i in xrange(n):
            action = self.env.random_action()
            self.perform_action(action)

    def noop_actions(self, n):
        # action 0 is NOOP
        for i in xrange(n):
            self.perform_action(0)



if __name__ == '__main__':

    environment_id = 'Pong-v0'
    minibatch_size = 32
    replay_memory_size = 35
    agent_history_length = 4
    target_network_update_frequency = 10000
    discount_factor = 0.99
    action_repeat = 4
    update_frequency = 4
    learning_rate = 0.00025
    gradient_momentum = 0.95
    squared_gradient_momentum = 0.95
    min_squared_gradient = 0.01
    initial_exploration = 1
    final_exploration = 0.1
    final_exploration_frame = 1000000
    replay_start_size = 50
    no_op_max = 30

    print 'starting replay memory ...'
    mem = ReplayMemory(replay_memory_size)
    print 'starting environment ...'
    env = Environment(environment_id)
    print 'starting model ...'
    model = Model(env.n_actions(), discount_factor, learning_rate)
    print 'starting agent ...'
    agent = Agent(model, env, mem, initial_exploration, final_exploration, final_exploration_frame, action_repeat)
    print 'populate memory with random actions ...'
    agent.random_actions(replay_start_size)
    for episode in xrange(1000000000):
        print 'episode', episode
        agent.env.reset()
        agent.noop_actions(random.randint(1, no_op_max))
        while not agent.env.is_done():
            agent.act()
            minibatch = agent.mem.sample(minibatch_size)
            agent.model.train(minibatch)
            # copy the model to the target model periodically
            if agent.frame % target_network_update_frequency == 0:
                agent.model.copy_model_to_target()
