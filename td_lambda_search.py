'''
CURRENT ISSUES
1. Append function is directly changing the state rather than a copy of the state
2. Error when checking input: expected dense_3_input to have 3 dimensions,
   but got array with shape (5, 1)
   https://stackoverflow.com/questions/44747343/keras-input-explanation-input-shape-units-batch-size-dim-etc

'''

import gym
from gym_blackjack import *
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation
import numpy as np

env = gym.make('blackjack-v0')
env.reset()

class TD_Lambda_Search():

	def __init__(self, env, gamma=0.9, lamb=0.9, alpha=0.9, epsilon=0.1, critic=None):

		import copy

		self.env = env
		self.clone = copy.deepcopy(env)
		self.gamma = gamma
		self.lamb = lamb
		self.alpha = alpha
		self.epsilon = epsilon

		self.action_space = [i for i in range(env.action_space.n)]

		# The Critic, with goal of estimating value function
		if critic is not None:
			self.critic = load_model(critic) # Load the critic passed to the class
		else:
			self.critic = Sequential() # Create the critic as a Tensorflow model
			self.critic.add(Dense(9, activation='relu', input_dim=5))
			self.critic.add(Dense(1, activation='sigmoid', bias_initializer='ones'))
			self.critic.compile(optimizer='SGD', loss='MSE', metrics=['accuracy'])

	def get_input(self, state, action):
		# Prepare the state and action for input to the Critic model
		import copy
		temp = np.append(copy.deepcopy(state), action).reshape(1,5)
		return temp

	def get_traces(self, replays, state=None):
		# Get the eligibility traces of each action in the replay buffer
		# Eligibility increases if the state has already been visited
		for row in reversed(range(len(replays))):
			replays[row][4] = (replays[row][4] * self.gamma * self.lamb) 
			if replays[row][0] == state:
				replays[row][4] += 1

		return replays

	def get_td_error(self, replay, this_value=None):

		_, next_value = self.greedy(replay[3]) # Choose the best hypothetical action
		if not this_value:
			this_value = self.critic.predict(self.get_input(replay[0], replay[1])) # Get value of action taken

		delta = self.alpha * (replay[2] + (self.gamma * next_value) - this_value)

		return delta

	def greedy(self, state, epsilon_greedy=False):
		# Takes the action with the highest expected return and its expected return
		# If epsilon_greedy=True, then (self.epsilon) percent of the time, take a random action

		action_dict = {1: self.critic.predict(self.get_input(state, 1)),
					   0: self.critic.predict(self.get_input(state, 0))}

		import random
		n = random.uniform(0, 1)
		if n > self.epsilon or not epsilon_greedy:
			action_to_take = max(action_dict, key=lambda x: action_dict[x])
		else:
			action_to_take = random.choice(list(action_dict.keys()))

		return action_to_take, action_dict[action_to_take]

	def one_simulation(self, max_steps):
		# Runs one planning simulation using a perfect model
		# (Clone of real environment + game rules)
		import copy
		self.clone = copy.deepcopy(self.env)
		self.clone.shuffle()

		replays = []

		for step in range(max_steps):
			print('one_simulation running', step)
			# Determine, and take, the appropriate step
			state = copy.deepcopy(self.clone.state)
			action, _ = self.greedy(self.clone.state, epsilon_greedy=True)

			obs, reward, done, info = self.clone.step(action)

			# Add the replay to the replay buffer, including a trace of 0 to start
			# get_traces() will turn the 0 to a 1 immediately
			replays.append([state, action, reward, obs, 0])
			replays = self.get_traces(replays, state)

			for row in replays:
				current_value = self.critic.predict(self.get_input(row[0], row[1]))
				updated_value = current_value + (self.get_td_error(row, current_value) * row[4])
				print("Values:", current_value, updated_value)
				self.critic.fit(self.get_input(row[0], row[1]), updated_value)

			if done:
				break

	def tree_search(self, k=10, max_steps=10):

		for i in range(k):
			self.one_simulation(max_steps)

	def one_game(self, k=10, max_steps=10):
		
		obs = self.env.state
		done = False
		while not done:
			# Train the model based on simulated games from this point foward
			self.tree_search(k, max_steps)	
			# Take the action the model thinks is greediest
			action, _ = self.greedy(obs)
			print(obs, action)
			obs, reward, done, info = self.env.step(action)

		print(f'Game over. Win/Lose = {reward}')
		print(f'Final Board: {obs}')
		self.env.reset()

if __name__ == "__main__":

	agent = TD_Lambda_Search(env)

	for i in range(1_000):
		agent.one_game()

	agent.critic.save("test_td_lambda_bias_ones")