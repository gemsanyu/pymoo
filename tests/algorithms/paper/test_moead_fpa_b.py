import sys

from tests.test_alg import test_alg
from pymoo.factory import get_reference_directions
from pymoo.algorithms.moead_fpa_b import MOEAD_ALFPA_B
from pymoo.decomposition.tchebicheff import Tchebicheff

n_runs = int(sys.argv[1])
problem_name = sys.argv[2]
report_file_name = sys.argv[3]
if len(sys.argv) > 4:
    n_var = int(sys.argv[4])
    n_obj = int(sys.argv[5])
else:
    n_obj = None
    n_var = None

print(n_runs, problem_name, report_file_name, n_var, n_obj)

if n_obj is None:
    ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=99)
else:
    ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=99)

algorithm = MOEAD_ALFPA_B(ref_dirs, decomposition=Tchebicheff())

test_alg(algorithm, n_runs, report_file_name, problem_name, n_var, n_obj)
