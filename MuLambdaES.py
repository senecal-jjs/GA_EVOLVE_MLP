import numpy as np
from collections import namedtuple


class MuLambda(Darwin):

    Individual = namedtuple('Individual', genes, sigmas)

    def create_instance(mu, lamb, pop_size, nodes_per_layer, actFunc, problem_type):
        obj = MuLambda(pop_size, nodes_per_layer, actFunc, problem_type)
        obj.mu = mu
        obj.lamb = lamb
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
            pop.append(Individual(genes=i,sigmas=sigmas))
        return pop

    def select_parents(self):
        raise "Don't call this method, silly"

    def evolve(self):
        pass
