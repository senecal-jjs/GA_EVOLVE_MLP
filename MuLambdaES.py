class MuLambda(Darwin):

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
        pass

    def select_parents(self):
        raise "Don't call this method, silly"

    def evolve(self):
        pass
