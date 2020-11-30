import numpy as np
from scipy.spatial.distance import cdist

from pymoo.algorithms.so_cuckoo_search import MantegnasAlgorithm
from pymoo.algorithms.genetic_algorithm import GeneticAlgorithm
from pymoo.decomposition.tchebicheff import Tchebicheff2
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.model.infill import InfillCriterion
from pymoo.factory import get_decomposition
from pymoo.model.individual import Individual
from pymoo.model.population import Population
from pymoo.operators.crossover.simulated_binary_crossover import SimulatedBinaryCrossover
from pymoo.operators.mutation.polynomial_mutation import PolynomialMutation
from pymoo.operators.sampling.random_sampling import FloatRandomSampling
from pymoo.util.display import MultiObjectiveDisplay
from pymoo.util.misc import set_if_none
from pymoo.operators.repair.to_bound import set_to_bounds_if_outside_by_problem


# =========================================================================================================
# Implementation
# =========================================================================================================
class Modified_RW(InfillCriterion):

    def __init__(self, alpha, beta, pa, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.levy = MantegnasAlgorithm(beta)
        self.mutation = PolynomialMutation(prob=None, eta=20)
        self.pa = pa

     #E is the neighbourhood in the form of indexes
    #i is the current solution to be modified
    def _grw(self, problem, X, E, i):
        #choose 1 random solution index from E
        r1 = np.random.choice(E, 1)[0]
        dvec = self.alpha*(X[r1] - X[i])*self.levy.do(problem.n_var)
        _x = X[i] + dvec
        _x = set_to_bounds_if_outside_by_problem(problem, _x[None, :])

        off = Population.new(X=_x)
        off = self.mutation.do(problem, off)
        return off

    def _lrw(self, problem, X, E, i):
        r1, r2 = np.random.choice(E, 2, replace=False)

        eps = np.random.rand()
        dvec = eps*(X[r1]-X[r2])
        if np.random.rand() > self.pa:
            dvec = 0
        _x = X[i] + dvec
        _x = set_to_bounds_if_outside_by_problem(problem, _x[None, :])

        off = Population.new(X=_x)
        return off

    def _do(self, problem, pop, n_offsprings, X, parents=None, **kwargs):
        if parents is None:
            raise Exception("parents cannot be None, must be [Neighbors, idx, op]")
        neighbors, idx, op = parents
        if op==0:
            off = self._grw(problem, X, neighbors, idx)
        else:
            off = self._lrw(problem, X, neighbors, idx)
        return off



class MOEAD_CS(GeneticAlgorithm):

    def __init__(self,
                 ref_dirs,
                 alpha=1,
                 beta=1.5,
                 c=5,
                 n_replacement=2,
                 n_neighbors=None,
                 decomposition=Tchebicheff2(),
                 pa=0.25,
                 prob_neighbor_mating=0.9,
                 display=MultiObjectiveDisplay(),
                 **kwargs):
        """

        MOEAD_CS Algorithm.

        Parameters
        ----------
        ref_dirs
        n_neighbors
        decomposition
        prob_neighbor_mating
        display
        kwargs
        """
        set_if_none(kwargs, 'pop_size', len(ref_dirs))
        set_if_none(kwargs, 'sampling', FloatRandomSampling())
        set_if_none(kwargs, 'survival', None)
        set_if_none(kwargs, 'selection', None)

        mating = Modified_RW(alpha=alpha, beta=beta, pa=pa)
        super().__init__(display=display, mating=mating, **kwargs)

        self.nds = NonDominatedSorting()
        self.ref_dirs = ref_dirs
        self.n_neighbors = n_neighbors
        if self.n_neighbors is None:
            self.n_neighbors = int(self.pop_size/10)
        self.n_replacement = n_replacement
        self.prob_neighbor_mating = prob_neighbor_mating
        self.decomposition = decomposition
        self.alpha = alpha
        self.levy = MantegnasAlgorithm(beta)

        #parameters for adaptive operators
        self.c = c
        self.W = int(self.pop_size/2)
        self.used_op = np.full(self.W, -1)
        self.op_rewards = np.full(self.W, 0)
        self.sw_idx = 0

        # neighbours includes the entry by itself intentionally for the survival method
        self.neighbors = np.argsort(cdist(self.ref_dirs, self.ref_dirs), axis=1, kind='quicksort')
        self.neighbors = self.neighbors[:, :self.n_neighbors]

    def _initialize(self):
        if isinstance(self.decomposition, str):
            decomp = self.decomposition
            self._decomposition = get_decomposition(decomp)
        else:
            self._decomposition = self.decomposition

        #init population and save ideal & nadir point
        super()._initialize()
        self.ideal_point = np.min(self.pop.get("F"), axis=0)
        pf = self.nds.do(self.pop.get("F"), only_non_dominated_front=True)
        self.nadir_point = np.max(pf, axis=0)
        self.max_angles = self._calc_max_angle(self.ref_dirs, self.pop.get("F"))

    def _calc_max_angle(self, w, F):
        f = F-self.ideal_point
        cos = np.dot(f, w.T) / np.dot(np.linalg.norm(f, axis=1), np.linalg.norm(w, axis=1))
        angles = np.arccos(cos)
        return np.max(angles, axis=1)

    #operator index : 0->Global Random Walk 1->Local Random Walk
    def _calc_op_score(self, FRR, op_freq):
        op_freq_ratio = self.c + np.sqrt(2*np.log(np.sum(op_freq))/op_freq)
        score = FRR + op_freq_ratio
        return score

    def _choose_op(self):
        I_op = [self.used_op == 0, self.used_op == 1]
        op_freq = np.array([len(self.used_op[I_op[0]]), len(self.used_op[I_op[1]])])
        #if there is an unused operator then randomly pick from the two
        if op_freq[0]==0 or op_freq[1]==0:
            return np.random.randint(2)
        reward0 = np.sum(self.op_rewards[I_op[0]])
        reward1 = np.sum(self.op_rewards[I_op[1]])
        tot_rewards = np.sum(self.op_rewards)
        FRR = np.array([reward0, reward1])
        op_score = self._calc_op_score(FRR, op_freq)
        return np.argmin(op_score)


    def _next(self):
        op = self._choose_op()
        X = self.pop.get("X")
        F = self.pop.get("F")
        for idx in range(self.pop_size):
            if np.random.rand() <= self.prob_neighbor_mating:
                E = self.neighbors[idx]
            else:
                E = np.arange(self.pop_size)

            #permute E and all its other corresponding values
            E = np.random.permutation(E)

            parents = np.array([E, idx, op], dtype='object')
            off = self.mating.do(self.problem, self.pop, 1, X=X, parents=parents, algorithm=self)
            self.evaluator.eval(self.problem, off)
            self.ideal_point = np.min(np.vstack([self.ideal_point, off.get("F")]), axis=0)

            max_angles_old = self.max_angles[E]
            w = self.ref_dirs[E]
            f = np.tile(off.get("F"), (len(E), 1))
            max_angles_new = self._calc_max_angle(w, f)

            FV = self._decomposition.do(F[E], weights=self.ref_dirs[E], ideal_point=self.ideal_point)
            off_FV = self._decomposition.do(f, weights=self.ref_dirs[E], ideal_point=self.ideal_point)


            #compute improved score and max angle difference
            G = (FV - off_FV)/FV
            DT = max_angles_old - max_angles_new
            Improved = np.logical_and(G>=0, DT>=0)
            # print(Improved)

            max_angles_new = max_angles_new[Improved]
            G = G[Improved]
            E = E[Improved]

            max_angles_new = max_angles_new[:self.n_replacement]
            G = G[:self.n_replacement]
            E = E[:self.n_replacement]

            F[E] = f[0]
            X[E] = off.get("X")[0]
            self.max_angles[E] = max_angles_new
            FIR = np.sum(G)

            #record used operator and the reward (improvement rate) obtained
            self.op_rewards[self.sw_idx] = FIR
            self.used_op[self.sw_idx] = op
            self.sw_idx = (self.sw_idx+1) % self.W

        self.pop.set("X", X)
        self.pop.set("F", F)
        #update nadir point
        pf = self.nds.do(F, only_non_dominated_front=True)
        self.nadir_point = np.max(pf, axis=0)

        #update max angles
        self.max_angles = self._calc_max_angle(self.ref_dirs, self.pop.get("F"))

# parse_doc_string(MOEAD_CS.__init__)
