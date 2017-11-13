from Darwin import Darwin
import numpy as np
import random

''' The DiffEvolution class contains the functionality to train a neural network using a differential evolution
    algorithm. This class is a child of the Darwin superclass and provides implementations of the evolve, select_parents, 
    crossover and mutate methods.'''


class DiffEvolution(Darwin):
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

	def evolve(self, validation_data):
		''' Combine selection, mutation, crossover, and replacement
			to evolve the population into its next generation '''

		new_pop = []

		#Create n offspring, where n is population size
		for i in range(len(self.population)):
			parents = self.select_parents()
			parent1 = parents[0]
			parent2 = self.mutate(parents[1:])
			offspring = self.binomial_crossover(parent1, parent2)
			self.parent_vs_offspring(parent1, offspring, validation_data)

	def parent_vs_offspring(self, parent, offspring, validation_data):
		''' Compare the parent and the offspring. The fitter of the
			two goes back into the population '''

		if self.fitness(offspring, validation_data) < self.fitness(parent, validation_data):
			self.population[0] = offspring

	def binomial_crossover(self, parent1, parent2):
		''' Given two parents, perform binomial crossover '''

		offspring = []
		for weight in parent1:
			offspring.append(weight)

		crossover_prob = 0.35 #This is tunable. 0.5 would result in uniform crossover
		j_star = random.randint(0, len(parent1))

		for j in range(len(parent1)):
			
			rand_prob = random.uniform(0, 1)

			if rand_prob < crossover_prob or j == j_star:
				offspring[j] = parent2[j]

		return offspring