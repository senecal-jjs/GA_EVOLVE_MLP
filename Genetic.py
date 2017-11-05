import numpy as np
import collections
import validation_funcs as validation
from Darwin import Darwin


class genetic_algorithm(Darwin):
    def create_instance(population_size, nodes_per_layer, activation_function, problem_type):
        obj = genetic_algorithm(population_size, nodes_per_layer, activation_function, problem_type)

        # Assign initial variance for mutation
        for i in range(len(obj.population)):
            obj.population[i].append(np.random.uniform(0, 1))
        return obj

    def train(self, num_iterations : int, training_data, validation_data):
        RMSE = []
        best_network = object
        best_rmse = 999

        # For number of specified generations evolve the network population
        for i in  range(num_iterations):
            if i % 5 == 0:
                # Calculate the rmse of the fittest individual in the population, and append to list of rmse at each
                # generation
                if self.problem.get() == "regression":
                    print("Beginning generation " + str(i) + " of " + str(num_iterations) + "...with rmse of: " + str(best_rmse))
                    if best_rmse < 2:
                        break
                elif self.problem.get() == "classification":
                    print("Beginning generation " + str(i) + " of " + str(num_iterations) + "...percent incorrect: " + str(best_rmse))
                    if best_rmse < 0.05: # 5% incorrect
                        break

                best_rmse = sys.maxsize
                for individual in self.population:
                    current_net = self.create_mlp(individual[0:-1])
                    current_rmse = validation.validate_network(current_net, validation_data, self.problem_type)

                    if current_rmse < best_rmse:
                        best_rmse = current_rmse
                        best_network = current_net

                RMSE.append(best_rmse)

            # GA parameter order: mutation rate, crossover rate, Num individuals for tournament, training data
            self.evolve(0.2, 0.8, 15, training_data)

        return best_network, RMSE


    def evolve(self, mutation_prob, crossover_prob, k, validation_data):
        np.random.shuffle(self.population)
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

        self.replace(offspring, "fittest", validation_data)

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
            selected_individuals.append(competitors[winner])

        return selected_individuals

    # Mutation using normal distribution
    def mutate(self, individual, mutation_prob):
        for i in range(len(individual) - 1):
            if np.random.uniform(0, 1) < mutation_prob:
                individual[i] += np.random.normal(0, 1)

    # One point crossover
    def crossover(self, ind1, ind2):
        pt = np.random.randint(0, len(ind1))

        child1 = ind1[:pt] + ind2[pt:len(ind2)]
        child2 = ind2[:pt] + ind1[pt:len(ind1)]

        return collections.namedtuple('children', ['child1', 'child2'])(child1, child2)
