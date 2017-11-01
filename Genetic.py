import numpy as np
import collections
from Darwin import Darwin
import time


class genetic_algorithm(Darwin):
    def create_instance(population_size, nodes_per_layer, activation_function, problem_type):
        obj = genetic_algorithm(population_size, nodes_per_layer, activation_function, problem_type)

        # Assign initial variance for self-adaptive mutation
        for i in range(len(obj.population)):
            obj.population[i].append(np.random.uniform(0, 5))  #[obj.population[i], np.random.uniform(0, 1)]
        return obj

    def evolve(self, mutation_prob, crossover_prob, k, validation_data):
        parents = self.select_parents(k, validation_data)
        offspring = []
        half_size = int(len(self.population)/2)

        for i in range(half_size):
            if np.random.uniform(0, 1) < crossover_prob:
                children = self.crossover(parents[i], parents[i + half_size])
            else:
                children = collections.namedtuple('children', ['child1', 'child2'])(parents[i], parents[i + half_size])

            self.mutate(children.child1, mutation_prob)
            self.mutate(children.child2, mutation_prob)

            offspring.append(children.child1)
            offspring.append(children.child2)

        self.replace(offspring, "generational")

    # Implementation of tournament based selection to choose parents, k = how many individuals compete in tournament
    def select_parents(self, k, validation_data):
        selected_individuals = []

        for i in range(self.population_size):
            competitors = []
            fitness = []

            for j in range(k):
                index = np.random.randint(0, self.population_size)
                competitors.append(self.population[index])
                fitness.append(self.fitness(self.population[index][0:-1], validation_data))

            winner = np.argmin(fitness)
            # if i % 10 == 0:
            #     print("Winner: " + str(competitors[winner]))
            selected_individuals.append(competitors[winner])

        return selected_individuals

    # Mutation using self-adaptive mutation strategy
    def mutate(self, individual, mutation_prob):
        for i in range(len(individual) - 1):
            if np.random.uniform(0, 1) < mutation_prob:
                u = np.random.normal(0, 1)
                individual[-1] = individual[-1] * np.exp(u / np.sqrt(len(individual[0:-1])))
                individual[i] += np.random.normal(0, 1) #np.random.normal(0, individual[-1])


    # Uniform crossover
    def crossover(self, ind1, ind2):
        # pt1 = np.random.randint(0, int(len(ind1[0])/2))
        # pt2 = np.random.randint(int(len(ind1[0])/2), len(ind1[0]))
        #
        # child1 = ind1[0][0:pt1] + ind2[0][pt1:pt2] + ind1[0][pt2:len(ind1[0])]
        # child2 = ind2[0][0:pt1] + ind1[0][pt1:pt2] + ind2[0][pt2:len(ind1[0])]

        mask = np.random.randint(0, 2, size=len(ind1))
        child1 = []
        child2 = []

        for i, bit in enumerate(mask):
            if bit == 0:
                child1.append(ind1[i])
                child2.append(ind2[i])
            else:
                child1.append(ind2[i])
                child2.append(ind1[i])

        return collections.namedtuple('children', ['child1', 'child2'])(child1, child2)
