import numpy as np
from scipy.spatial.distance import cdist

from pymoo.operators.mutation.polynomial_mutation import PolynomialMutation
from pymoo.algorithms.so_cuckoo_search import MantegnasAlgorithm
from pymoo.algorithms.nsga2 import RankAndCrowdingSurvival
from pymoo.algorithms.so_fpa import FlowerPollinationAlgorithm
from pymoo.decomposition.pbi import PBI
from pymoo.docs import parse_doc_string
from pymoo.model.population import Population
from pymoo.model.individual import Individual
from pymoo.operators.repair.to_bound import set_to_bounds_if_outside
from pymoo.operators.sampling.latin_hypercube_sampling import LHS
from pymoo.util.reference_direction import sample_on_unit_simplex
from pymoo.util.display import MultiObjectiveDisplay
from pymoo.util.misc import set_if_none
from pymoo.util.termination.default import MultiObjectiveDefaultTermination
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


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
    def _do(self, xi, xr1, xr2):
        #find n_offsprings*2 different solutions (n_offsprings pair)
        _x = xi + 0.1*(xr1-xr2)
        return _x

class MOEAD_ALFPA(FlowerPollinationAlgorithm):

    def __init__(self,
                 ref_dirs,
                 alpha=0.1,
                 c=5,
                 decomposition=PBI(),
                 n_neighbors=None,
                 p=0.9,
                 sampling=LHS(),
                 termination=None,
                 display=MultiObjectiveDisplay(),
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

        set_if_none(kwargs, 'pop_size', len(ref_dirs))
        set_if_none(kwargs, 'survival', None)
        set_if_none(kwargs, 'selection', None)

        super().__init__(sampling=sampling,
                         termination=termination,
                         display=display,
                         mating=None,
                         **kwargs)

        cauchy = AdaptiveLevyFlight(alpha, 1)
        gaussian = AdaptiveLevyFlight(alpha, 2)
        levy1 = AdaptiveLevyFlight(alpha, 1.3)
        # levy2 = AdaptiveLevyFlight(alpha, 1.5)
        levy3 = AdaptiveLevyFlight(alpha, 1.7)
        lrw = LocalPollination()

        #5 operators
        self.mating = [cauchy, gaussian, levy1, levy3, lrw]
        self.mutation = PolynomialMutation(prob=None, eta=20)
        self.n_op = len(self.mating)

        self.n_neighbors = n_neighbors
        if n_neighbors is None:
            #set to 20% of pop_size
            self.n_neighbors = int(self.pop_size/5)

        self.prob_neighbor_mating = p
        self.decomposition = decomposition
        self.ref_dirs = ref_dirs

        if self.ref_dirs.shape[0] < self.n_neighbors:
            print("Setting number of neighbours to population size: %s" % self.ref_dirs.shape[0])
            self.n_neighbors = self.ref_dirs.shape[0]

        # neighbours includes the entry by itself intentionally for the survival method
        self.neighbors = np.argsort(cdist(self.ref_dirs, self.ref_dirs), axis=1, kind='quicksort')[:, :self.n_neighbors]

        #parameters for adaptive operators
        self.c = c
        self.W = int(self.pop_size/2)
        self.used_op = np.full(self.W, -1)
        self.op_rewards = np.full(self.W, 0)
        self.sw_idx = 0
        self.op_mask = np.tile(np.arange(len(self.mating)), (self.W, 1)).T

    def _initialize(self):
        super()._initialize()
        self.ideal_point = np.min(self.pop.get("F"), axis=0)

    def _choose_op(self):
        current_mask = (self.used_op == self.op_mask).astype(dtype='float')
        op_freq = np.sum(current_mask, axis=1)
        if np.any(op_freq == 0):
            unused_op = np.where(op_freq == 0)[0]
            return unused_op[np.random.randint(len(unused_op))]
        tot_rewards = np.sum(self.op_rewards)
        FRR = np.dot(current_mask, self.op_rewards)/tot_rewards
        op_score = FRR + (self.c + np.sqrt(2*np.log(np.sum(op_freq))/op_freq))
        return np.argmin(op_score)

    def _set_optimum(self):
        I = NonDominatedSorting().do(self.pop.get("F"), only_non_dominated_front=True)
        self.opt = self.pop[I]

    def _next(self):
        X = self.pop.get("X")
        F = self.pop.get("F")
        CV = self.pop.get("CV")
        feasible = self.pop.get("feasible")
        xl, xu = self.problem.bounds()
        offs = Population()
        idx_permutation = np.random.permutation(self.pop_size)

        for i in idx_permutation:
            # parents, choose between neighbors or whole population
            N = self.neighbors[i]
            if np.random.rand() < self.prob_neighbor_mating:
                parents = np.random.choice(N, 2, replace=False)
            else:
                parents = np.random.choice(self.pop_size, 2, replace=False)

            #currently just randomly pick operator
            op = self._choose_op()
            if op<self.n_op-1:
                _x = self.mating[op]._do(X[parents[0]], X[i], xl, xu)
            else:
                _x = self.mating[op]._do(X[i], X[parents[0]], X[parents[1]])

            # evaluate the offspring
            # _x = set_to_bounds_if_outside(_x, xl, xu)
            _x = self.mutation._do(self.problem, _x[None, :])
            off = Individual(X=_x[0])
            self.evaluator.eval(self.problem, off)

            self.ideal_point = np.minimum(self.ideal_point, off.F)

            # calculate the decomposed values for each neighbor
            FV = self.decomposition.do(F[N], weights=self.ref_dirs[N, :], ideal_point=self.ideal_point)
            off_FV = self.decomposition.do(off.F[None, :], weights=self.ref_dirs[N, :], ideal_point=self.ideal_point)

            # get the absolute index in F where offspring is better than the current F (decomposed space)
            I = np.where(off_FV < FV)[0]
            X[N[I]] = off.X
            F[N[I]] = off.F
            CV[N[I]] = off.CV
            feasible[N[I]] = off.feasible

            #compute improved score and max angle difference
            G = (FV - off_FV)/FV
            FIR = np.sum(G[I])

            #record used operator and the reward (improvement rate) obtained
            self.op_rewards[self.sw_idx] = FIR
            self.used_op[self.sw_idx] = op
            self.sw_idx = (self.sw_idx+1) % self.W

        self.pop = Population.new(X=X, F=F, CV=CV, feasible=feasible)

parse_doc_string(MOEAD_ALFPA.__init__)
