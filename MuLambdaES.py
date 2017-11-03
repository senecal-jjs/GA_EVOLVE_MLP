import numpy as np
import math
from collections import namedtuple


class MuLambda(Darwin):

    Individual = namedtuple('Individual', genes, sigmas)

    def create_instance(mu, lamb, pop_size, nodes_per_layer, actFunc, problem_type):
        obj = MuLambda(pop_size, nodes_per_layer, actFunc, problem_type)
        obj.mu = mu
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
            pop.append(MuLambda.Individual(genes=i,sigmas=sigmas))
        return pop

    def _loc_new_sigma(indiv):
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
        new_pop = self.population.copy()
        for indiv in np.random.choice(self.population, self.lamb, replace=True):
            new_pop.append(self.mutate(indiv))
        self.population = self._pick_top(self.population_size, new_pop, validation_data)

    def global_es(self, validation_data):
        pass

    def evolve(self, variance_type, validation_data):
        if variance_type == "local":
            self.local_es(validation_data)
        else:
            self.global_es(validation_data)
        t = t + 1

    def select_parents(self):
        raise "Don't call this method, silly"
