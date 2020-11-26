import numpy as np

from pymoo.algorithms.so_cuckoo_search import MantegnasAlgorithm
from pymoo.algorithms.so_genetic_algorithm import FitnessSurvival, GeneticAlgorithm
from pymoo.docs import parse_doc_string
from pymoo.model.infill import InfillCriterion
from pymoo.model.population import Population
from pymoo.model.replacement import ImprovementReplacement
from pymoo.operators.repair.to_bound import ToBoundOutOfBoundsRepair
from pymoo.operators.sampling.latin_hypercube_sampling import LHS
from pymoo.util.display import SingleObjectiveDisplay
from pymoo.util.termination.default import SingleObjectiveDefaultTermination


# =========================================================================================================
# Pollination
# =========================================================================================================
class Pollination(InfillCriterion):

    def __init__(self, alpha, beta, p, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.levy = MantegnasAlgorithm(beta)
        self.p = p

    def _do(self, problem, pop, n_offsprings, parents=None, **kwargs):
        if parents is None:
            raise Exception("For pollination please provide the parents!")

        X, F = pop.get("X", "F")
        xl, xu = problem.bounds()

        # a = original index [1..pop_size]
        # b = repeated index of best solution
        # c,d = pair of random different solutions' indexes
        a, b, c, d = parents

        # get random levy values to be used for the global pollination step size
        # levy = np.random.normal(0, 1, size=(len(parents), problem.n_var))
        global_step = self.levy.do(size=(len(a), problem.n_var))
        #global pollination direction
        global_direction = self.alpha*(xu-xl)*global_step*(X[b] - X[a])

        # get random value from uniform distribution for the local step size
        # repeat for n_var
        local_step = np.random.rand(len(a))
        local_step = np.tile(local_step, (problem.n_var, 1)).T
        #local pollination direction
        local_direction = local_step*(X[c]-X[d])

        #determine which solutions do local or global pollination via heaviside based on given p
        H = (np.random.rand(len(a)) < self.p).astype(dtype='float')
        H = np.tile(H, (problem.n_var, 1)).T

        _X = X[a] + H*global_direction + (1-H)*local_direction
        _X = ToBoundOutOfBoundsRepair().do(problem, _X)
        # _X = InversePenaltyOutOfBoundsRepair().do(problem, _X, P=X[a])

        return Population.new(X=_X, index=a)

# =========================================================================================================
# Implementation
# =========================================================================================================
class FlowerPollinationAlgorithm(GeneticAlgorithm):

    def __init__(self,
                 pop_size=25,
                 beta=1.5,
                 alpha=0.01,
                 p=0.8,
                 mating=None,
                 sampling=LHS(),
                 termination=SingleObjectiveDefaultTermination(),
                 display=SingleObjectiveDisplay(),
                 **kwargs):
        """

        Parameters
        ----------

        sampling : {sampling}

        termination : {termination}

        pop_size : int
         The number of nests to be used

        beta : float
            The input parameter of the Mantegna's Algorithm to simulate
            sampling on Levy Distribution

        alpha : float
            The step size scaling factor and is usually 0.01.

        pa : float
            The switch probability, pa fraction of the nests will be abandoned on every iteration
        """
        mating = kwargs.get("mating")
        if mating is None:
            mating = Pollination(alpha, beta, p)

        super().__init__(pop_size=pop_size,
                         sampling=sampling,
                         termination=termination,
                         display=display,
                         mating=mating,
                         **kwargs)


    def _next(self):
        pop = self.pop
        best = FitnessSurvival().do(self.problem, pop, 1, return_indices=True)[0]

        #population indexes, all solutions will be mutated
        a = np.arange(len(pop))
        #best solutions g* for the global pollination
        b = np.repeat(best, len(pop))

        # randomly select 2 different solutions for the local pollination
        rand_pair = np.argpartition(np.random.rand(len(pop), len(pop)), 2, axis=1)[:,:2]
        c,d = rand_pair.T


        P = np.stack([a,b,c,d])
        # do the flower pollination and evaluate the result offsprings
        off = self.mating.do(self.problem, pop, len(pop), parents=P, algorithm=self)
        self.evaluator.eval(self.problem, off, algorithm=self)

        # replace the old solution with offspring if offspring is better
        has_improved = ImprovementReplacement().do(self.problem, pop, off, return_indices=True)

        # replace the individuals in the population
        self.pop[has_improved] = off[has_improved]
        self.off = off


parse_doc_string(FlowerPollinationAlgorithm.__init__)
