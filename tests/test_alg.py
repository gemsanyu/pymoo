import csv
import numpy as np

from pymoo.factory import get_problem
from pymoo.optimize import minimize
from pymoo.performance_indicator.igd import IGD
from pymoo.performance_indicator.hv import Hypervolume


hv_ref_point = np.array([1.0, 1.0])
n_pareto_points = 1000

#test function
def test_alg(algorithm, n_runs, report_file_name, problem_name, n_var=None, n_obj=None):
    with open(report_file_name, mode='a') as result_file:
        writer = csv.writer(result_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        # writer.writerow(["problem_name", "n_var", "n_obj", "n_eval",
        #                  "mean_rt", "std_rt",
        #                  "mean_igd", "std_igd", "min_igd", "max_igd",
        #                  "mean_hv", "std_hv", "min_hv", "max_hv"])

        if n_var is not None:
            if n_obj is not None:
                problem = get_problem(problem_name, n_var=n_var, n_obj=n_obj)
            else:
                problem = get_problem(problem_name, n_var=n_var)
        else:
            problem = get_problem(problem_name)

        igd_vals = np.array([])
        hv_vals = np.array([])
        exec_times = np.array([])

        try:
            pf = problem.pareto_front(n_pareto_points=n_pareto_points)
        except:
            pf = problem.pareto_front(n_points=n_pareto_points)

        for r in range(n_runs):
            res = minimize(problem, algorithm, ("n_eval", 25000), verbose=False)
            igd_val = IGD(pf=pf, normalize=False).calc(res.F)
            igd_vals = np.append(igd_vals, igd_val)

            hv_val = Hypervolume(ref_point=hv_ref_point, normalize=True, pf=pf).calc(res.F)
            hv_vals = np.append(hv_vals, hv_val)

            exec_times = np.append(exec_times, res.exec_time)

        writer.writerow([problem_name, problem.n_var, problem.n_obj, 25000,
                     np.mean(exec_times), np.std(exec_times),
                     np.mean(igd_vals), np.std(igd_vals), np.min(igd_vals), np.max(igd_vals),
                     np.mean(hv_vals), np.std(hv_vals), np.min(hv_vals), np.max(hv_vals)] )
        result_file.flush()
