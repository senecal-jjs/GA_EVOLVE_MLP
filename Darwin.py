from abc import ABC, abstractmethod
from operator import itemgetter
import MLP
import numpy as np

''' Parent class of Genetic, EvolutionStrategy, and DiffEvolution. Defines the mutate, select_parents, evolve, and 
    crossover methods. The fitness function is implemented here as well as methods to create a population of networks. '''

class Darwin(ABC):
	def __init__(self, population_size, nodes_per_layer, activation_function, problem_type):
		self.population_size = population_size
		self.nodes_per_layer = nodes_per_layer
		self.activation_function = activation_function
		self.problem_type = problem_type
		self.population = self._create_population()

	@abstractmethod
	def mutate(self):
		''' Abstract method for mutating the offspring '''
		pass

	@abstractmethod
	def select_parents(self):
		''' Abstract method for selecting the parents to use
			for reproduction '''
		pass

	@abstractmethod
	def evolve(self):
		''' Abstract method for the evolution (mutation and 
			crossover) of the population '''
		pass

	def crossover(self, ind1, ind2):
		''' Given two individuals, perform uniform crossover
			to produce a new individual '''

		mask = np.random.randint(0, 2, size = len(ind1))
		new = []

		for i,bit in enumerate(mask):

			if bit == 0:
				new.append(ind1[i])
			if bit == 1:
				new.append(ind2[i])

		return new

	def fitness(self, individual, validation_data):
		# Using 0-1 loss for classification, and RMSE for regression, test the fitness of an individual
	    # in the population

		# Convert the individual to a network
		net = self.create_mlp(individual)

		# Calculate the average error over the training data
		summed_error = 0
		for instance in validation_data:
			if net.problem_type == "regression":
				network_output = net.calculate_outputs(instance.inputs)
				summed_error += np.sqrt((network_output - instance.solution)**2)
			elif net.problem_type == "classification":
				network_output = net.calculate_outputs(instance.inputs)
				predicted_index = np.argmax(network_output)
				true_index = np.argmax(instance.solution)

				if predicted_index != true_index:
					summed_error += 1

		return summed_error/len(validation_data)

	# Method to replace the previous generation with a new generation
	def replace(self, offspring, method, validation_data):
		# Given the current population, the offspring, and the
		# method for replacement, create the new population

		if method == "fittest":
			#Select the n fittest from both population and offspring
			n = len(self.population)
			pool = self.population + offspring
			pool_fitness = []

			for individual in pool:
				pool_fitness.append((individual, self.fitness(individual, validation_data)))

			pool_fitness = sorted(pool_fitness, key=itemgetter(1))
			temp_population = []
			for i in range(len(self.population)): temp_population.append(pool_fitness[i][0])
			self.population = temp_population

		elif method == "generational":
			#Replace the old generation with the new
			self.population = offspring

		elif method == "steady":
			#Replace some parents and some offspring
			pass

	def _create_population(self):
		# Create the initial population to be evolved.

		pop = []
		for i in range(self.population_size):

			vector_size = 0

			for j in range(len(self.nodes_per_layer) - 1):
				vector_size += (self.nodes_per_layer[j] + 1) * self.nodes_per_layer[j+1]
			
			pop.append(np.random.normal(0, 2, size = vector_size).tolist())

		return pop

	def create_mlp(self, individual):
		# Using the weights in the population, create an
		# MLP network to test on

		net = MLP.network(self.nodes_per_layer, self.activation_function, self.problem_type)

		x = 0
		for i,layer in enumerate(net.layers):

			if i == len(self.nodes_per_layer) - 1:
				break

			for j in range(self.nodes_per_layer[i] + 1):

				for k in range(self.nodes_per_layer[i + 1]):

					layer.weights[j][k] = individual[x]
					x += 1

		return net

