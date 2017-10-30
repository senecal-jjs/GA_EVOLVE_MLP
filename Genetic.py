import numpy as np
import collections
from Darwin import Darwin


class genetic_algorithm(Darwin):
    def create_instance(population_size, nodes_per_layer, activation_function, problem_type):
        obj = genetic_algorithm(population_size, nodes_per_layer, activation_function, problem_type)

        # Assign initial variance for self-adaptive mutation
        for i in range(len(obj.population)):
            obj.population[i] = [obj.population[i], np.random.uniform(0, 0.5)]
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
                fitness.append(self.fitness(self.population[index][0], validation_data))

            winner = np.argmin(fitness)
            selected_individuals.append(competitors[winner])

        return selected_individuals

    # Mutation using self-adaptive mutation strategy
    def mutate(self, individual, mutation_prob):
        if np.random.uniform(0, 1) < mutation_prob:
            u = np.random.normal(0, 1)
            individual[1] = individual[1] * np.exp(u/np.sqrt(len(individual[0])))

            for i in range(len(individual[0])):
                individual[0][i] += np.random.normal(0, individual[1])

    # Uniform crossover
    def crossover(self, ind1, ind2):
        mask = np.random.randint(0, 2, size=len(ind1[0]))
        child1 = []
        child2 = []

        for i, bit in enumerate(mask):
            if bit == 0:
                child1.append(ind1[0][i])
                child2.append(ind2[0][i])
            else:
                child1.append(ind2[0][i])
                child2.append(ind1[0][i])

        child1 = [child1, ind1[1]]
        child2 = [child2, ind2[1]]

        return collections.namedtuple('children', ['child1', 'child2'])(child1, child2)
