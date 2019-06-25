import tensorflow as tf
import gym
from gym.spaces import Discrete, Box
import numpy as np


class Reinforce():

    def __init__(self, environment, hidden_layers, lr=0.01, clip_norm=False):

        self.env = environment # check if has discrete actions
        self.obs_shape = self.env.observation_space.shape[0]
        self.actions_shape = self.env.action_space.n

        # check environment
        assert isinstance(self.env.observation_space, Box), \
            "This example only works for envs with continuous state spaces."
        assert isinstance(self.env.action_space, Discrete), \
            "This example only works for envs with discrete action spaces."

        # build model
        input = tf.keras.layers.Input(shape=(self.obs_shape,))
        hidden = input
        for h in hidden_layers:
            hidden = tf.keras.layers.Dense(h, activation='relu')(hidden)

        output = tf.keras.layers.Dense(self.actions_shape, activation='softmax')(hidden)

        self.model = tf.keras.Model(inputs=input, outputs=output)

        # define optimizer
        self.optimizer = tf.keras.optimizers.Adam(lr)
        self.clipping = clip_norm

        # logging
        self.logger = {'average_return' : 0,
                       'gradient_step' : 0}

    def train_epoch(self, num_trajectories=4, render=True):

        # collect data
        trajectories = [self.collect_trajectory(render)
                        for t in range(num_trajectories)] # renders last trajectory

        states, actions, returns, total_rewards = zip(*trajectories)
        states, actions, returns = np.vstack(states), np.hstack(actions), np.hstack(returns)

        # perform gradient update and logging
        self.logger['gradient_step'] = self._gradient_update(states, actions, returns, num_trajectories)
        self.logger['average_return'] = np.average(total_rewards)

    #@tf.function - yells at me for some reason
    def _gradient_update(self, states, actions, returns, num_trajectories):

        with tf.GradientTape() as tape:
            probs = tf.boolean_mask(self.model(states), tf.one_hot(actions,self.actions_shape))
            target = -1/num_trajectories * tf.reduce_sum(tf.math.log(probs) * returns) # use minus and use gradient descent

        gradients = tape.gradient(target, self.model.variables)
        if self.clipping:
            gradients = [tf.clip_by_value(grad, -self.clipping, self.clipping)
                         for grad in gradients]

        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables)) # gradient descent update

        return tf.linalg.global_norm(gradients)

    def collect_trajectory(self, render=False):

        s, a ,r = [], [], []
        done = False
        state = self.env.reset()
        while not done:
            if render: self.env.render()

            policy = self.policy(state)[0]
            action = np.random.choice(self.actions_shape, 1, p=policy)[0] # sample action from policy
            obs, reward, done, info = env.step(action)

            if done: reward = 0

            s.append(state)
            a.append(action)
            r.append(reward)

            state = obs

        total_rewards = np.sum(r)
        return np.vstack(s), np.array(a), self.reward_to_go(r), total_rewards

    @staticmethod
    def reward_to_go(r):
        return np.cumsum(r[::-1])[::-1]

    def policy(self, state):
        return self.model.predict(np.atleast_2d(state))

    def print_log(self):
        for attr, item in self.logger.items():
            print('{} : {}'.format(attr, item))

        print('----------------------------')

if __name__ == '__main__':
    env = gym.make('CartPole-v0')

    agent = Reinforce(env, [32], clip_norm=False)#, logdir=logdir)
    epochs, trajectories = 100, 32 # trajectories per epoch. Ie batch size

    for epoch in range(epochs):
        agent.train_epoch(32, render=False)
        agent.print_log()
