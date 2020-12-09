import numpy as np

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
from pymoo.operators.sampling.latin_hypercube_sampling import LHS
from pymoo.util.reference_direction import sample_on_unit_simplex
from pymoo.util.display import MultiObjectiveDisplay
from pymoo.util.termination.default import MultiObjectiveDefaultTermination

# =========================================================================================================
# Implementation
# =========================================================================================================
class AdaptiveLevyFlight:
    def __init__(self, alpha, beta):
        self.alpha = alpha
        #beta must be in range of [1..2]
        beta = min(beta, 2)
        beta = max(beta, 1)
        self.beta = beta
        self.cauchy = np.random.default_rng().standard_cauchy
        self.gaussian = np.random.default_rng().standard_normal
        self.levy = MantegnasAlgorithm(beta)

    def _do(self, xr, xi, xl, xu):
        # get random levy/cauchy/gaussian values to be used for the step size
        if self.beta == 1:
            levy = self.cauchy(len(xi))
        elif self.beta == 2:
            levy = self.gaussian(len(xi))
        else:
            levy = self.levy.do(len(xi))
        direction = (xr-xi)
        _x = xi + (xu - xl)*self.alpha * levy * direction
        return _x

class LocalPollination:
    def _do(self, X, xi, xl, xu, n_offsprings):
        #find n_offsprings*2 different solutions (n_offsprings pair)
        Pair = np.random.permutation(X)[:2*n_offsprings]
        R1, R2 = Pair[:n_offsprings], Pair[n_offsprings:2*n_offsprings]
        _X = xi + 0.01*(R1-R2)
        return _X

class MO_ALFPA(FlowerPollinationAlgorithm):

    def __init__(self,
                 pop_size=100,
                 alpha=0.1,
                 mating=None,
                 p=0.8,
                 sampling=LHS(),
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
        levy1 = AdaptiveLevyFlight(alpha, 1.3)
        levy2 = AdaptiveLevyFlight(alpha, 1.7)
        lrw = LocalPollination()
        self.mating = [cauchy, gaussian, levy1, levy2, lrw]
        self.mutation = PolynomialMutation(prob=None, eta=10)
        self.index = np.arange(self.pop_size)

        self.p0 = p

    def setup(self, problem, **kwargs):
        super().setup(problem, **kwargs)
        #estimate max_gen from max_eval
        self.max_gen = int(self.n_max_evals/self.pop_size)


    def _next(self):

        # updating p
        self.p = self.p0 - 0.1 * ((self.max_gen-self.n_gen) / self.max_gen)

        X = self.pop.get("X")
        F = self.pop.get("F")
        rank = self.pop.get("rank")
        xl, xu = self.problem.bounds()
        offs = Population()
        index_permute = np.random.permutation(self.index)[:int(self.pop_size/4)]
        for i in index_permute:
            xi = X[i]
            #pick operator GRW or LRW
            #GRW = operators 1-4
            #LRW = operator 5 with n_offsprings=4
            _X = []
            if np.random.rand() <= self.p:
                r = np.random.randint(self.pop_size)
                xr = np.random.permutation(X[rank==0])[0]
                for op in range(4):
                    _x = self.mating[op]._do(xr, xi, xl, xu)
                    _X = _X + [_x]
            else:
                _X = self.mating[4]._do(X, xi, xl, xu, 4)
            _X = self.mutation._do(self.problem, np.array(_X))
            off = Population.new(X=_X)
            offs = Population.merge(offs, off)
        # replace the individuals in the population
        self.evaluator.eval(self.problem, offs, algorithm=self)
        self.pop = Population.merge(self.pop, offs)
        self.pop = self.survival._do(self.problem, self.pop, self.pop_size)



parse_doc_string(MO_ALFPA.__init__)
