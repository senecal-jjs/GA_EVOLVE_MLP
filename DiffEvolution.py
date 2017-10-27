from Darwin import Darwin
import numpy as np
import MLP
import random

class DiffEvolution(Darwin):

	#def __init__(self, beta):
	#	self.beta = beta

	def create_instance(beta, population_size, nodes_per_layer):
		obj = DiffEvolution(population_size, nodes_per_layer)
		obj.beta = beta
		return obj 

	def mutate(self, individuals):
		''' Given three individuals, create the trial vector, u
			to cross with another individual '''

		print ("Mutating...")
		trial_vector = individuals[0]
		return trial_vector

	def select_parents(self):
		''' Select 4 random parents for mutating and crossover'''

		random.shuffle(self.population)
		parents = self.population[0:4]
		print ("PARENTS: " + str(parents))
		return parents

	def evolve(self):
		''' Combine selection, mutation, crossover, and replacement
			to evolve the population into its next generation '''

		print ("Evolving...")

		new_pop = []

		#Create n offspring, where n is population size
		#Each loop through creates two offspring, so do n/2 loops
		for i in range(len(self.population)):
			parents = self.select_parents()
			parent1 = parents[0]
			parent2 = self.mutate(parents[1:])
			print("parent 1: " + str(parent1) + "   parent 2: " + str(parent2))
			offspring = self.crossover(parent1, parent2)
			new_pop.append(offspring1)
			#new_pop.append(offspring2)

		self.replace(new_pop, "fittest")


if __name__ == '__main__':
	test = DiffEvolution.create_instance(0.2, 5, [2, 3, 1])
	for i,ind in enumerate(test.population):
		print ("Individual " + str(i) + ": " + str(ind))