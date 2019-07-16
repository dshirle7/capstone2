# MONTE CARLO SEARCH

# Reinforcement learning algorithm that uses observed win rate to determine actions
# Necessarily converges on the optimal strategy given enough time
# Tabular method

import gym
from gym_blackjack import *
import numpy as np

from tqdm import tqdm

env = gym.make('blackjack-v0')
env.reset()

class Monte_Carlo_Search():

	def __init__(self, env, epsilon=0.2, value_fn_path=None):

		import copy
		import pandas as pd

		self.env = env
		self.clone = copy.deepcopy(env)
		self.epsilon = epsilon

		self.action_space = [i for i in range(env.action_space.n)]

		if value_fn_path is not None:
			self.value_fn = pd.read_csv(value_fn_path)
			print(self.value_fn)

		else:
			from itertools import product

			states_list = list(product(range(12,22), range(2,11), range(2), [0]))
			states_aces = list(product(range(12,22), [11], range(2), [1]))

			states_list.extend(states_aces)
			
			self.value_fn = pd.DataFrame(data=states_list, columns=['Player', 'Dealer', 'P Ace', 'D Ace'])
			self.value_fn['State'] = self.value_fn.apply(lambda x: tuple(x), axis=1)
			self.value_fn.drop(['Player', 'Dealer', 'P Ace', 'D Ace'], axis=1, inplace=True)
			# Initialize Hit and Stay with 100% win rates.
			self.value_fn['1_win'] = 1
			self.value_fn['1_count'] = 1
			self.value_fn['0_win'] = 1
			self.value_fn['0_count'] = 1


	def greedy(self, state, epsilon_greedy=False):
		# POLICY:
		# Take the action that has the best chance of winning given the state
		# If epsilon_greedy=True, then (self.epsilon) percent of the time, take a random action

		action_dict = {}

		for action in self.action_space:
			wins = self.value_fn.loc[self.value_fn['State'] == state, str(action) + '_win']
			count = self.value_fn.loc[self.value_fn['State'] == state, str(action) + '_count']
			action_dict[action] = int(wins) / int(count)

		import random
		n = random.uniform(0, 1)
		if n > self.epsilon or not epsilon_greedy:
			action_to_take = max(action_dict, key=action_dict.get)
		else:
			action_to_take = random.choice(list(action_dict.keys()))

		return action_to_take

	def one_simulation(self):
		# Runs one planning simulation using a perfect model
		# (Clone of real environment + game rules)

		import copy
		self.clone = copy.deepcopy(self.env)
		self.clone.shuffle()

		# First planning step is epsilon-greedy to ensure the entire action space is searched
		# Every successive step is only greedy

		state = tuple(self.clone.state)
		action = self.greedy(state, epsilon_greedy=True)

		state_actions = [(tuple(state), action)]

		obs, reward, done, info = self.clone.step(action)

		while not done:
			# Proceed with the simulation greedily until the game is over
			state = tuple(self.clone.state)
			action = self.greedy(state, epsilon_greedy=False)
			state_actions.append((state, action))
			obs, reward, done, info = self.clone.step(action)

# BUG: Agent is ONLY recording counts, not wins, even if it does in fact win.

		for pair in state_actions:
			state, action = pair
			if reward == 1 or reward == 0.5:
				# If you just won or pushed the game, increment the win count and the play count
				self.value_fn.loc[self.value_fn['State'] == state, str(action) + '_win'] += reward
				self.value_fn.loc[self.value_fn['State'] == state, str(action) + '_count'] += 1
			elif reward == 0:
				# If you just lost the game, increment the play count only
				self.value_fn.loc[self.value_fn['State'] == state, str(action) + '_count'] += 1
			else:
				raise ValueError('reward is an unexpected number')

	def tree_search(self, k=100):
		for i in range(k):
			self.one_simulation()

	def one_game(self, k=100):

		obs = tuple(self.env.state)
		done = False
		while not done:
			self.tree_search(k)
			action = self.greedy(tuple(obs))
			obs, reward, done, info = self.env.step(action)

#		print("Final State:", obs, reward, done, info)
#		if reward == 1:
#			print("Agent won!")
#		if reward == 0.5:
#			print("Agent pushed.")
#		if reward == 0:
#			print("Agent lost.")
		self.env.reset()

if __name__ == "__main__":

	filename = "test_monte_carlo.csv"

	try:
		agent = Monte_Carlo_Search(env, value_fn_path=filename)
	except OSError:
		agent = Monte_Carlo_Search(env)

	for i in tqdm(range(10_000)):
		agent.one_game()

	agent.value_fn.to_csv(filename, index=False)