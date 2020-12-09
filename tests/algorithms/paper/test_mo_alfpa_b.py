import sys

from tests.test_alg import test_alg
from pymoo.algorithms.mo_alfpa_b import MO_ALFPA_B

n_runs = int(sys.argv[1])
problem_name = sys.argv[2]
report_file_name = sys.argv[3]
if len(sys.argv) > 4:
    n_var = int(sys.argv[4])
    if len(sys.argv) > 5:
        n_obj = int(sys.argv[5])
    else:
        n_obj = None
else:
    n_obj = None
    n_var = None

print(n_runs, problem_name, report_file_name, n_var, n_obj)
algorithm = MO_ALFPA_B()

test_alg(algorithm, n_runs, report_file_name, problem_name, n_var, n_obj)
