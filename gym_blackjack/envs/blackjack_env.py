import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import random

class BlackjackEnv(gym.Env):
	# Blackjack environment that runs on a single deck of cards
	metadata = {'render.modes': ['human']}

	def __init__(self):

		self.action_space = spaces.Discrete(2)
		# Observation Space:
		# 	1. Player's value (assuming aces are 11)
		#	2. Dealer's shown value
		#	3. Whether the player's ace could count as a 1 or 11 
		# 	4. Whether the dealer's ace could count as a 1 or 11
		self.observation_space = spaces.MultiDiscrete([10, 10, 2, 2])

		self.state = None
		self.done = False
		self.steps_beyond_done = None

		self.deck = None

	def step(self, action):

		# Convert hits to stays if the player is already at 21
		if action == 0 or self.state[0] == 21:
			reward = self.stay()

		if action == 1:
			self.hit()
			reward = 0

		if not self.done:
			pass
		elif self.steps_beyond_done is None:
			# The episode is over
			self.steps_beyond_done = 0
		else:
			if self.steps_beyond_done == 0:
				pass
#				logger.warn("You are calling 'step' even though this environment has already returned done = True")
			self.steps_beyond_done += 1
			reward = 0

		return np.array(self.state), reward, self.done, {}

	def reset(self):
		self.done = False

		self.deck = [i for i in range(2, 10)] * 4
		self.deck.extend([10] * 16)
		self.deck.extend([11] * 4)
		random.shuffle(self.deck)
		self.state = [0] * 4

		self.hit(1)
		while self.state[0] < 12:
			self.hit(0)

		return np.array(self.state)

	def shuffle(self):
		random.shuffle(self.deck)

	def render(self, mode='human'):
		print(self.state)

	def hit(self, player=0):
		# Deals the next card to the appropriate position

		if player not in [0, 1]:
			raise ValueError(f'hit() can only handle player of 0 or 1; received {player}')

		# If the player gets an ace, show it in the state
		if self.deck[-1] == 11:
			self.state[player + 2] = 1

		# Add the card's value to the player's current total
		self.state[player] += self.deck.pop()

		if self.state[player] > 21 and self.state[player + 2] == 0:
			# The player just went bust
			self.done = True
		elif self.state[player] > 21 and self.state[player + 2] == 1:
			# The player must use their soft ace as a 1
			self.state[player] += -10
			self.state[player + 2] = 0
		else:
			pass

	def stay(self):
		# Agent has chosen to stay
		# This function represents dealer's behavior afterwards
		# Then it evaluates the state and determines who wins

		self.done = True

		while self.state[1] < 17:
			self.hit(1)

		if self.state[1] > 21:
			reward = 1
			return reward
		elif self.state[0] > self.state[1]:
			reward = 1
			return reward
		elif self.state[0] == self.state[1]:
			reward = 0.5
			return reward

		reward = 0
		return reward