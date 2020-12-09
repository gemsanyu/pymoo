import numpy as np
from scipy.spatial.distance import cdist

from pymoo.algorithms.mo_alfpa import LocalPollination, AdaptiveLevyFlight
from pymoo.operators.mutation.polynomial_mutation import PolynomialMutation
from pymoo.algorithms.so_fpa import MantegnasAlgorithm
from pymoo.algorithms.nsga2 import RankAndCrowdingSurvival
from pymoo.algorithms.so_fpa import FlowerPollinationAlgorithm
from pymoo.decomposition.weighted_sum import WeightedSum
from pymoo.docs import parse_doc_string
from pymoo.model.infill import InfillCriterion
from pymoo.model.population import Population
from pymoo.model.replacement import ImprovementReplacement
from pymoo.operators.repair.to_bound import set_to_bounds_if_outside
from pymoo.operators.sampling.random_sampling import FloatRandomSampling
from pymoo.util.reference_direction import sample_on_unit_simplex
from pymoo.util.display import MultiObjectiveDisplay
from pymoo.util.termination.default import MultiObjectiveDefaultTermination

class MO_ALFPA_B(FlowerPollinationAlgorithm):

    def __init__(self,
                 pop_size=100,
                 alpha=0.1,
                 mating=None,
                 sampling=FloatRandomSampling(),
                 k=3,
                 termination=None,
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

        cauchy = AdaptiveLevyFlight(alpha, 1)
        gaussian = AdaptiveLevyFlight(alpha, 2)
        levy = AdaptiveLevyFlight(alpha, 1.5)
        # levy1 = AdaptiveLevyFlight(alpha, 1.3)
        # levy2 = AdaptiveLevyFlight(alpha, 1.7)
        lrw = LocalPollination()
        self.mating = [[gaussian, lrw], [cauchy, levy]]
        self.mutation = PolynomialMutation(prob=None, eta=10)
        self.index = np.arange(self.pop_size)

        self.k = k

    def setup(self, problem, **kwargs):
        super().setup(problem, **kwargs)

    def _pick_op(self, i, Rank, HAD):
        if Rank[i] == 0:
            pool = 0
        else:
            pool = 1

        j = np.random.randint(self.pop_size)
        if HAD[i]<HAD[j]:
            op = 0
        elif HAD[i]>HAD[j]:
            op = 1
        else:
            op = np.random.randint(2)

        return pool, op

    def _calc_HAD(self, F, k=3):
        distances =  np.sort(cdist(F, F), axis=1, kind='quicksort')[:, :k]
        distances[distances==0] = np.inf
        distances = 1/distances
        return k/np.sum(distances, axis=1)

    def _next(self):

        X = self.pop.get("X")
        F = self.pop.get("F")
        HAD = self._calc_HAD(F, self.k)
        rank = self.pop.get("rank")
        xl, xu = self.problem.bounds()
        offs = Population()
        for i in self.index:
            xi = X[i]
            _X = []
            pool, op = self._pick_op(i, rank, HAD)
            opr = self.mating[pool][op]
            if opr.type=="grw":
                xr = np.random.permutation(X[rank==0])[0]
                _X = opr._do(xr, xi, xl, xu)[None, :]
            else:
                _X = opr._do(X, xi, xl, xu, 1)
            _X = self.mutation._do(self.problem, np.array(_X))
            off = Population.new(X=_X)
            offs = Population.merge(offs, off)
        # replace the individuals in the population
        self.evaluator.eval(self.problem, offs, algorithm=self)
        self.pop = Population.merge(self.pop, offs)
        self.pop = self.survival._do(self.problem, self.pop, self.pop_size)

parse_doc_string(MO_ALFPA_B.__init__)
