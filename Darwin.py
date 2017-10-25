from abc import ABC, abstractmethod
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
			to produce two new individuals '''

		mask = np.random.randint(0, 2, size = len(ind1))
		print ("Mask: " + str(mask))
		new1 = []
		new2 = []
		i = 0
		for bit in mask:

			if bit == 0:
				new1.append(ind1[i])
				new2.append(ind2[i])

			if bit == 1:
				new1.append(ind2[i])
				new2.append(ind1[i])

			i+=1

		return new1, new2

	def fitness(self, individual):
		''' Using 0-1 loss, test the fitness of an individual 
			in the population '''

		pass

	def replace(self):
		''' Replace an indiviual back into the population '''

		pass

	def _create_population(self):
		''' Create the initial population to be evolved. '''
		
		pop = [] #change this to the wrapper class data type in the network class
		for i in range(self.population_size):
			for j in range(len(self.nodes_per_layer) - 1):
				#Add one to the number of nodes to account for bias nodes
				weight_layer = np.random.uniform(-0.2, 0.2, 
					size=((self.nodes_per_layer[j] + 1) * self.nodes_per_layer[j+1]))
				pop.append(weight_layer)
		return pop