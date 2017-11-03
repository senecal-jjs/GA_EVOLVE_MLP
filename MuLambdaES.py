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

        return obj

    def mutate(self, individuals):
        "Return one mutated individual"
        pass

    # need to overload this to get paired sigma values:
    def _create_population(self):
        pop = []
        items = super._create_population()
        for i in items:
            # add the sigma values
            sigmas = np.random.uniform(0,1, len(i))
            pop.append(MuLambda.Individual(genes=i,sigmas=sigmas))
        return pop

    def select_parents(self):
        raise "Don't call this method, silly"

    def _loc_new_sigma(indiv):
        return indiv.sigmas * np.exp(u_zero * self.r1 + self.r2 * np.random.normal(0,1))

    def _local_es(self):
        pass

    def _global_se(self):
        pass

    def evolve(self, variance_type):
        if variance_type == "local":
            self._local_es()
        else:
            self._global_se()
