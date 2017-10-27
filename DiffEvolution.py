from Darwin import Darwin
import numpy as np
import MLP
import random

class DiffEvolution(Darwin):

	#def __init__(self, beta):
	#	self.beta = beta

	def create_instance(beta, population_size, nodes_per_layer, activation_function, problem_type):
		obj = DiffEvolution(population_size, nodes_per_layer, activation_function, problem_type)
		obj.beta = beta
		return obj 

	def mutate(self, individuals):
		''' Given three individuals, create the trial vector, u
			to cross with another individual '''

		ind1 = np.array(individuals[0])
		ind2 = np.array(individuals[1])
		ind3 = np.array(individuals[2])

		trial_vector = np.add(ind1, (self.beta * np.subtract(ind2, ind3)))

		return trial_vector.tolist()

	def select_parents(self):
		''' Select 4 random parents for mutating and crossover'''

		random.shuffle(self.population)
		parents = self.population[0:4]
		return parents

	def evolve(self):
		''' Combine selection, mutation, crossover, and replacement
			to evolve the population into its next generation '''

		new_pop = []

		#Create n offspring, where n is population size
		#Each loop through creates two offspring, so do n/2 loops
		for i in range(len(self.population)):
			parents = self.select_parents()
			parent1 = parents[0]
			parent2 = self.mutate(parents[1:])
			offspring = self.crossover(parent1, parent2)
			new_pop.append(offspring)
			#new_pop.append(offspring2)

		self.replace(new_pop, "fittest")


if __name__ == '__main__':
	test = DiffEvolution.create_instance(0.1, 4, [2, 2, 1], "sigmoid", "classification")
	#for i,ind in enumerate(test.population):
	#	print ("Individual " + str(i) + ": " + str(ind))
	net = test.create_mlp(test.population[0])
	for layer in net.layers:
		print (layer.weights)