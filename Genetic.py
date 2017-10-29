import numpy as np
from collections import namedtuple
from Darwin import Darwin


class genetic_algorithm(Darwin):
    def create_instance(population_size, nodes_per_layer, activation_function, problem_type):
        obj = genetic_algorithm(population_size, nodes_per_layer, activation_function, problem_type)

        # Assign initial variance for self-adaptive mutation
        individual = namedtuple('individual', ['chromosome', 'stdev'])
        for i in range(len(obj.population)):
            obj.population[i] = individual(obj.population[i], np.random.uniform(-0.01, 0.01))
        return obj

    def evolve(self, mutation_prob, crossover_prob, k):
        parents = self.select_parents(k)
        offspring = []

        for i in range(len(self.population)):
            child = parents[i]

            if np.random.uniform(0, 1) < crossover_prob and (i+1) < len(self.population):
                child = self.crossover(parents[i], parents[i+1])

            if np.random.uniform(0, 1) < mutation_prob:
                self.mutate(child)

            offspring.append(child)

        self.replace(offspring, "generational")

    # Implementation of tournament based selection to choose parents, k = how many individuals compete in tournament
    def select_parents(self, k):
        selected_individuals = []

        for i in range(self.population_size):
            competitors = []
            fitness = []

            for j in range(k):
                index = np.random.randint(0, self.population_size)
                competitors.append(self.population[index])
                fitness.append(self.fitness(self.population[index].chromosome))

            winner = np.argmax(fitness)
            selected_individuals.append(competitors[winner])

        return selected_individuals

    # Mutation using self-adaptive mutation strategy
    def mutate(self, individual):
        u = np.random.normal(0, 1)
        individual.stdev = individual.stdev * np.exp(u/np.sqrt(len(individual.chromosome)))

        for gene in individual.chromosome:
            gene += np.random.normal(0, individual.stdev)
