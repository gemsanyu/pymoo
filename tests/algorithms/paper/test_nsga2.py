#test nsga2
from tests.test_alg import test_alg
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.operators.crossover.simulated_binary_crossover import SimulatedBinaryCrossover
from pymoo.operators.mutation.polynomial_mutation import PolynomialMutation

sbx = SimulatedBinaryCrossover(eta=20, prob=0.9)
pm = PolynomialMutation(prob=None, eta=20)
algorithm = NSGA2(crossover=sbx, mutation=pm)

test_alg(algorithm, 20, "result/nsga2.csv")
