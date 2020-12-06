#test nsga2
from tests.test_alg import test_alg
from pymoo.factory import get_reference_directions
from pymoo.algorithms.moead_fpa import MOEAD_ALFPA
from pymoo.decomposition.tchebicheff import Tchebicheff

n_obj = 2
ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=99)
algorithm = MOEAD_ALFPA(ref_dirs, decomposition=Tchebicheff())

test_alg(algorithm, 20, "moead_alfpa_tch_357.csv")
