import sys

from tests.test_alg import test_alg
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.operators.crossover.simulated_binary_crossover import SimulatedBinaryCrossover
from pymoo.operators.mutation.polynomial_mutation import PolynomialMutation

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

sbx = SimulatedBinaryCrossover(eta=20, prob=0.9)
pm = PolynomialMutation(prob=None, eta=20)
algorithm = NSGA2(crossover=sbx, mutation=pm)

test_alg(algorithm, n_runs, report_file_name, problem_name, n_var, n_obj)
