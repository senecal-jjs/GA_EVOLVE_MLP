from Darwin import Darwin
import numpy as np
import MLP

class DiffEvolution(Darwin):

	def __init__(self, beta):
		self.beta = beta

	def mutate(self):
		''' Given three individuals, create the trial vector, u
			to cross with another individual '''

		print ("Mutating...")

	def select_parents(self):
		''' Select the parents for mutating '''

		print ("Selecting parents...")

	def evolve(self):
		''' Combine selection, mutation, crossover, and replacement
			to evolve the population into its next generation '''

		print ("Evolving...")

if __name__ == '__main__':
	test = DiffEvolution(0.5)
	test.mutate()
	test.select_parents()
	test.evolve()