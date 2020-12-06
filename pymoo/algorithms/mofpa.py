import numpy as np

from pymoo.algorithms.nsga2 import RankAndCrowdingSurvival
from pymoo.algorithms.so_fpa import FlowerPollinationAlgorithm
from pymoo.decomposition.weighted_sum import WeightedSum
from pymoo.docs import parse_doc_string
from pymoo.model.infill import InfillCriterion
from pymoo.model.population import Population
from pymoo.model.replacement import ImprovementReplacement
from pymoo.operators.repair.to_bound import ToBoundOutOfBoundsRepair
from pymoo.operators.sampling.latin_hypercube_sampling import LHS
from pymoo.util.reference_direction import sample_on_unit_simplex
from pymoo.util.display import MultiObjectiveDisplay
from pymoo.util.termination.default import MultiObjectiveDefaultTermination

# =========================================================================================================
# Implementation
# =========================================================================================================
class MOFPA(FlowerPollinationAlgorithm):

    def __init__(self,
                 pop_size=100,
                 beta=1.5,
                 alpha=0.1,
                 p=0.8,
                 mating=None,
                 sampling=LHS(),
                 termination=MultiObjectiveDefaultTermination(),
                 display=MultiObjectiveDisplay(),
                 survival=RankAndCrowdingSurvival(),
                 **kwargs):
        """

        Parameters
        ----------

        sampling : {sampling}

        termination : {termination}

        pop_size : int
         The number of solutions

        beta : float
            The input parameter of the Mantegna's Algorithm to simulate
            sampling on Levy Distribution

        alpha : float
            The step size scaling factor and is usually 0.1.

        pa : float
            The switch probability, pa fraction of the nests will be abandoned on every iteration
        """
        super().__init__(pop_size=pop_size,
                         sampling=sampling,
                         termination=termination,
                         display=display,
                         mating=mating,
                         survival=survival,
                         **kwargs)

    def _next(self):
        pop = self.pop
        best_indexes = np.arange(len(pop))[pop.get("rank") == 0]

        #population indexes, all solutions will be mutated
        a = np.arange(len(pop))
        #best solutions g* for the global pollination
        b = np.random.choice(best_indexes, len(pop))

        # randomly select 2 different solutions for the local pollination
        rand_pair = np.argpartition(np.random.rand(len(pop), len(pop)), 2, axis=1)[:,:2]
        c, d = rand_pair.T

        P = np.stack([a,b,c,d])
        # do the flower pollination and evaluate the result offsprings
        off = self.mating.do(self.problem, pop, len(pop), parents=P, algorithm=self)
        self.evaluator.eval(self.problem, off, algorithm=self)

        # replace the individuals in the population
        self.pop = Population.merge(self.pop, off)
        self.pop = self.survival._do(self.problem, self.pop, self.pop_size)

parse_doc_string(MOFPA.__init__)
