from abc import ABC, abstractmethod
from operator import itemgetter
import MLP
import numpy as np

class Darwin(ABC):
	#__metaclass__ = ABCMeta

	def __init__(self, population_size, nodes_per_layer):
		self.population_size = population_size
		self.nodes_per_layer = nodes_per_layer
		self.population = self._create_population()

	@abstractmethod
	def mutate(self):
		''' Abstract method for mutating the offspring '''
		pass

	@abstractmethod
	def select_parents(self):
		''' Abstract method for selecting the partents to use
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
		#new2 = []

		for i,bit in enumerate(mask):

			if bit == 0:
				new1.append(ind1[i])
				#new2.append(ind2[i])

			if bit == 1:
				new1.append(ind2[i])
				#new2.append(ind1[i])

		return new #, new2

	def fitness(self, individual):
		''' Using 0-1 loss, test the fitness of an individual 
			in the population '''

		return individual[0] + individual[1] #Change this to 0-1 loss

	def replace(self, offspring, method):
		''' Given the current population, the offspring, and the 
			method for replacement, create the new population '''

		if method == "fittest":
			#Select the n fittest from both population and offspring
			n = len(self.population)
			pool = self.population + offspring
			pool_fitness = []

			for individual in pool:
				pool_fitness.append((individual, self.fitness(individual)))

			pool_fitness = sorted(pool_fitness, key=itemgetter(1))
			self.population = [ind[0] for ind in pool_fitness[:n:-1]]

		elif method == "generational":
			#Replace the old generation with the new
			self.population = offspring

		elif method == "steady":
			#Replace some parents and some offspring
			pass

	def _create_population(self):
		''' Create the initial population to be evolved. '''
		
		'''pop = [] #change this to the wrapper class data type in the network class
		for i in range(self.population_size):
			for j in range(len(self.nodes_per_layer) - 1):
				#Add one to the number of nodes to account for bias nodes
				weight_layer = np.random.uniform(-0.2, 0.2, 
					size=((self.nodes_per_layer[j] + 1) * self.nodes_per_layer[j+1]))
				pop.append(weight_layer)
		return pop'''

		pop = []
		for i in range(self.population_size):

			vector_size = 0

			for j in range(len(self.nodes_per_layer) - 1):
				vector_size += (self.nodes_per_layer[j] + 1) * self.nodes_per_layer[j+1]
			
			pop.append(np.random.uniform(-0.2, 0.2, size = vector_size))

		return pop