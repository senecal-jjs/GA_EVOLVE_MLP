import numpy as np
import math
import Darwin
from collections import namedtuple


class EvolutionStrategy(Darwin):
    """Population for EvolutionStrategy is different from the superclass:
    each individual is a namedtuple(genes, sigmas).
    """

    Individual = namedtuple('Individual', ['genes', 'sigmas'])

    def create_instance(lamb, pop_size, nodes_per_layer, actFunc, problem_type):
        obj = EvolutionStrategy(pop_size, nodes_per_layer, actFunc, problem_type)
        obj.lamb = lamb
        # for the local varience:
        # r values chosen based off of Back and Schwefel (1993)
        obj.r1 = 1/math.sqrt(2 * len(obj.population[0]))
        obj.r2 = 1/math.sqrt(2 * math.sqrt(len(obj.population[0])))
        obj.t = 0
        return obj

    # need to overload this to get paired sigma values:
    def _create_population(self):
        pop = []
        items = super._create_population()
        for i in items:
            # add the sigma values
            # initial sigma suggested to be ~3.0 (Thomas Back, 1996)
            sigmas = np.ones(len(i)) * 3.0
            pop.append(EvolutionStrategy.Individual(genes=i,sigmas=sigmas))
        return pop

    def _loc_new_sigma(indiv):
        """Return a set of new sigma values for an individual. Doesn't update
        the individual. Suggested by Computational Intelligence Second edition, Section 13.2.3
        """
        return indiv.sigmas * np.exp(u_zero * self.r1 + self.r2 * np.random.normal(0,1))

    def mutate(self, individual):
        "Returns a new individual that has been mutated based on the given individual"
        new_genes = []
        new_sigmas = self._loc_new_sigma(indiv)
        for g, sig in zip(individual.genes, new_sigmas):
            # no ceiling or floor for our weights:
            new_genes.append(g + np.random.normal(0, sig))
        return Individual(sigmas=new_sigmas, genes=new_genes)

    def _pick_top(n, canidates, validation_data):
        pool = []
        for indiv in canidates:
            pool.append((indiv, self.fitness(indiv.genes, validation_data)))
        pool.sort(key=itemgetter(1))
        return [ind[0] for ind in pool[:n-1:-1] ]

    def local_es(self, validation_data):
        """ Performs one iteration of the Local variance adaptation variation
        of the evolution stratagy algorithm
        """
        new_pop = self.population.copy()
        for indiv in np.random.choice(self.population, self.lamb, replace=True):
            new_pop.append(self.mutate(indiv))
        self.population = self._pick_top(self.population_size, new_pop, validation_data)

    def global_es(self, validation_data):
        """ Performs one iteration of the global variance adaptation variation
        of the evolution stratagy algorithm
        """
        pass

    def evolve(self, validation_data, variance_type="local"):
        """ Performs one iteration of an addaptive version of the evolution strategy
        algorithm. Specify "local" or "global" to choose what type of sigma update to
        use. Uses "local" by default.
        """
        if variance_type == "local":
            self.local_es(validation_data)
        else:
            self.global_es(validation_data)
        t = t + 1

    def train(self, num_iterations, training_data, validation_data):
        """Trains the network.
        """
        for i in range(num_iterations):
            for ind in self.population:
                self.evolve()

    def select_parents(self):
        """ This function is not needed for the current version of the algorithm, and
        will raise an error when called.
        """
        raise "Don't call this method, silly"
