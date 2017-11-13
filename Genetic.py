import numpy as np
import collections
from Darwin import Darwin

''' The genetic_algorithm class contains the functionality to train a neural network using a genetic algorithm.
    This class is a child of the Darwin superclass and provides implementations of the evolve, select_parents, 
    crossover and mutate methods.'''

class genetic_algorithm(Darwin):
    def create_instance(population_size, nodes_per_layer, activation_function, problem_type):
        obj = genetic_algorithm(population_size, nodes_per_layer, activation_function, problem_type)

        # Assign initial variance for mutation
        for i in range(len(obj.population)):
            obj.population[i].append(np.random.uniform(0, 1))
        return obj

    # Perform steps of evolution (select parents, then perform crossover, then mutate).
    def evolve(self, mutation_prob, crossover_prob, k, validation_data):
        np.random.shuffle(self.population)

        # Select parents
        parents = self.select_parents(k, validation_data)
        offspring = []
        half_size = int(len(self.population)/2)

        # Perform crossover
        for i in range(half_size):
            if np.random.uniform(0, 1) < crossover_prob:
                children = self.crossover(parents[i], parents[i + half_size])
            else:
                children = collections.namedtuple('children', ['child1', 'child2'])(parents[i], parents[i + half_size])

            # Mutate children
            self.mutate(children.child1, mutation_prob)
            self.mutate(children.child2, mutation_prob)

            offspring.append(children.child1)
            offspring.append(children.child2)

        # Create the next generation using the fittest individuals from the parents and children
        self.replace(offspring, "fittest", validation_data)

    # Implementation of tournament based selection to choose parents, k = how many individuals compete in tournament
    def select_parents(self, k, validation_data):
        selected_individuals = []

        # Perform |population| tournaments
        for i in range(self.population_size):
            competitors = []
            fitness = []

            # Randomly select k individuals to compete in tournament
            for j in range(k):
                index = np.random.randint(0, self.population_size)
                competitors.append(self.population[index])
                fitness.append(self.fitness(self.population[index][0:-1], validation_data))

            # Select winner of the tournament
            winner = np.argmin(fitness)
            selected_individuals.append(competitors[winner])

        return selected_individuals

    # Mutation using normal distribution
    def mutate(self, individual, mutation_prob):
        for i in range(len(individual) - 1):
            if np.random.uniform(0, 1) < mutation_prob:
                individual[i] += np.random.normal(0, 0.5)

    # One point crossover
    def crossover(self, ind1, ind2):
        pt = np.random.randint(0, len(ind1))

        child1 = ind1[:pt] + ind2[pt:len(ind2)]
        child2 = ind2[:pt] + ind1[pt:len(ind1)]

        return collections.namedtuple('children', ['child1', 'child2'])(child1, child2)
