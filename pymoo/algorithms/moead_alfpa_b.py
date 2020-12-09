import numpy as np
from scipy.spatial.distance import cdist

from pymoo.algorithms.mo_alfpa import AdaptiveLevyFlight
from pymoo.operators.mutation.polynomial_mutation import PolynomialMutation
from pymoo.algorithms.nsga2 import RankAndCrowdingSurvival
from pymoo.algorithms.so_fpa import FlowerPollinationAlgorithm
from pymoo.decomposition.tchebicheff import Tchebicheff
from pymoo.docs import parse_doc_string
from pymoo.model.population import Population
from pymoo.model.individual import Individual
from pymoo.operators.repair.to_bound import set_to_bounds_if_outside
from pymoo.operators.sampling.random_sampling import FloatRandomSampling
from pymoo.util.reference_direction import sample_on_unit_simplex
from pymoo.util.display import MultiObjectiveDisplay
from pymoo.util.misc import set_if_none
from pymoo.util.termination.default import MultiObjectiveDefaultTermination
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

class LocalPollination:
    def __init__(self):
        super().__init__()
        self.type = "lrw"

    def _do(self, xi, xa, xb):
        _x = xi + 0.01*(xa-xb)
        return _x

class MOEAD_ALFPA_B(FlowerPollinationAlgorithm):

    def __init__(self,
                 ref_dirs,
                 alpha=0.1,
                 c=5,
                 decomposition=Tchebicheff(),
                 n_neighbors=None,
                 n_replacement=None,
                 p=0.8,
                 sampling=FloatRandomSampling(),
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
        # levy1 = AdaptiveLevyFlight(alpha, 1.3)
        levy2 = AdaptiveLevyFlight(alpha, 1.5)
        # levy3 = AdaptiveLevyFlight(alpha, 1.7)
        lrw = LocalPollination()

        #5 operators
        self.mating = [[lrw, gaussian], [cauchy, levy2]]
        self.mutation = PolynomialMutation(prob=None, eta=20)

        self.n_neighbors = n_neighbors
        if n_neighbors is None:
            #set to 20% of pop_size
            self.n_neighbors = int(self.pop_size/10)

        self.prob_neighbor_mating = p
        self.decomposition = decomposition
        self.ref_dirs = ref_dirs

        if self.ref_dirs.shape[0] < self.n_neighbors:
            print("Setting number of neighbours to population size: %s" % self.ref_dirs.shape[0])
            self.n_neighbors = self.ref_dirs.shape[0]

        # neighbours includes the entry by itself intentionally for the survival method
        self.neighbors = np.argsort(cdist(self.ref_dirs, self.ref_dirs), axis=1, kind='quicksort')[:, :self.n_neighbors]
        self.n_replacement = n_replacement

        #saving the ranking
        self.rank = np.ones(self.pop_size)


    def _initialize(self):
        super()._initialize()
        self.ideal_point = np.min(self.pop.get("F"), axis=0)
        I = NonDominatedSorting().do(self.pop.get("F"), only_non_dominated_front=True)
        self.rank[I] = 0

    def _calc_HAD(self, x, arr):
        distances = cdist(x[None, :], arr)
        distances[distances==0] = np.inf
        distances = 1/distances
        return len(arr)/np.sum(distances)

    def _pick_op(self, F, i):
        #get its rank first
        rank = self.rank[i]
        if rank == 0:
            pool = 0
        else:
            pool = 1

        #get one random solution from Population
        j = np.random.randint(self.pop_size)

        #get their neighbors
        Ni = self.neighbors[i]
        Nj = self.neighbors[j]

        #calculate their Harmonic Average Distance
        #F[Ni[1:]] so that F[i] is not counted as its neighbor
        HADi = self._calc_HAD(F[i], F[Ni[1:]])
        HADj = self._calc_HAD(F[j], F[Nj[1:]])

        #solution i is less crowded than j
        if HADi > HADj:
            op = 1
        #solution i is more crowder than j
        elif HADi < HADj:
            op = 0
        #else just randomise
        else:
            op = np.random.randint(2)

        return pool, op


    def _get_domination(self, f, F):
        #check if f is dominated by any of the F
        not_better = np.all(F<=f, axis=1)
        has_worse = np.any(F<f, axis=1)
        is_dominated = np.any(np.logical_and(not_better, has_worse))
        if is_dominated:
            rank = 1
        else:
            rank = 1

        not_worse = np.all(f<=F, axis=1)
        has_better = np.any(f<F, axis=1)
        domination_index = np.logical_and(not_worse, has_better)

        return rank, domination_index


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
            pool, op = self._pick_op(F, i)
            opr = self.mating[pool][op]
            if opr.type=="grw":
                xr = X[parents[0]]
                _x = opr._do(xr, X[i], xl, xu)
            else:
                _x = opr._do(X[i], X[parents[0]], X[parents[1]])
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
            #if the new solution is going to replace at least 1 solution
            if len(I) > 0:
                #then update the rank
                rank, domination_index = self._get_domination(off.F, F)
                self.rank[domination_index] = 1

                #then replace the solution
                replaced_index = N[I]
                if self.n_replacement is not None:
                    replaced_index = replaced_index[:self.n_replacement]
                X[replaced_index] = off.X
                F[replaced_index] = off.F
                CV[replaced_index] = off.CV
                feasible[replaced_index] = off.feasible
                self.rank[replaced_index] = rank

        self.pop = Population.new(X=X, F=F, CV=CV, feasible=feasible)

parse_doc_string(MOEAD_ALFPA_B.__init__)
