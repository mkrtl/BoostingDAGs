"""
This script runs experiments for the algorithm BoostingDAGs. 
It takes a path to a directory containing data and true graphs.
"""
import logging
import multiprocessing as mp
import time
import os

from sklearn.gaussian_process import kernels

from boostdags.causal_discovery import DAGBooster
from boostdags.pruning import gam_pruning
from simulation.graph import evaluate_graphs, write_edges_to_csv, load_data_and_graph
import constants

logging.basicConfig(level=logging.INFO)

# Number of experiments
n_exps = 10
# Number of variables, only needed to find experiment directories
p = 100
# Number of samples, only needed to find experiment directories
N = 200
# Are the structural equations additive?
additive = True
# Use scaled version of the data? Only needed to find experiment directories
scale_data = True
scaled_string = "scaled" if scale_data else "unscaled"
# Stopping criterion for the boosting algorithm
m = "AIC"
# m = dict(leave_out_ration=0.5)
# Erdos-Renyi graph (ER) or Barabasi-Albert graph (BA)?
graph_type = "ER"
# Number of cores to use
ncores = 1
# Step size boosting
mu = 0.3
# Regularization parameter RKHS regression
alpha = 0.01
# Save result to csv file in the experiment directory?
save_results = False
# Name of result file in target directory?
target_file_name = "boosting_cam_edges.csv"
# Run experiment even if <target_file_name> already exists in target directory?
run_again = True

# path_data = f"C:/Users/Max/Desktop/BoostingDAGs/Experiments/dens_edges_100.00/{scaled_string}/graph_{p}_{N}_additive_{additive}_{graph_type}"
# path_data = os.path.join(os.path.abspath(path_data), f"{mu}_{alpha}")
path_data = constants.PATH_DATA_LARGE_P(scaled_string, p, N, additive, graph_type)


def run_experiment(
    mu=0.3,
    alpha=0.01,
    m="AIC",
    p=100,
    path_data=path_data,
    store_result_edges=True,
    run_again=False,
):
    kernel_func_estimator = kernels.RBF(1.0)
    target_path = f"{path_data}/{target_file_name}"
    if os.path.exists(target_path) and run_again is False:
        logging.info(f"Experiment {path_data} already exists.")
        return None
    data, true_graph = load_data_and_graph(path_data)
    edges = true_graph.edges
    logging.info(f"Running experiment {path_data}...")
    logging.info(f"The true graph has {len(true_graph.edges)} edges.")

    dag_boost = DAGBooster(
        kernel_func_estimator,
        p,
        m=m,
        mu=mu,
        alpha=alpha
    )
    
    start = time.time()
    dag_boost.train(data)
    dag_boost_after_pruning = dag_boost.pruning(gam_pruning, data)
    end = time.time()
    duration = end - start
    evaluation_result = evaluate_graphs(dag_boost_after_pruning, true_graph)
    shd = evaluation_result["SHD"]

    n_true_positives = evaluation_result["n_true_positives"]
    n_false_positives = evaluation_result["n_false_positives"]
    n_false_negatives = evaluation_result["n_false_negatives"]
    logging.info(f"The structural hamming distance is {shd}")

    if store_result_edges:
        write_edges_to_csv(
            dag_boost_after_pruning, path_data, file_name=target_file_name
        )
    return dict(
        SHD=shd,
        n_true_positives=n_true_positives,
        n_false_positives=n_false_positives,
        n_false_negatives=n_false_negatives,
        n_edges_true_graph=len(true_graph.edges),
        duration=duration,
    )


def parallel_experiment_execution(
    mu=0.3,
    alpha=0.01,
    m="AIC",
    p=100,
    ncores=None,
    store_result_edges=True, 
    run_again=False,
    n_exps=n_exps,
    path_data=path_data
):
    paths_data = [os.path.join(os.path.abspath(path_data), p) for p in os.listdir(path_data) if os.path.isdir(os.path.join(path_data, p))]
    paths_data = paths_data[:n_exps]
    if ncores is None:
        ncores = mp.cpu_count()
    pool = mp.Pool(ncores)
    with pool:
        processes = [
            pool.apply_async(
                run_experiment,
                kwds=dict(
                    mu=mu,
                    alpha=alpha,
                    m=m,
                    p=p,
                    path_data=path_data,
                    store_result_edges=store_result_edges,
                    run_again=run_again
                ),
            )
            for path_data in paths_data
        ]
        result = [p.get() for p in processes]

    return result


if __name__ == "__main__":
 
    print(f"Scale data: {scale_data}")
    result = parallel_experiment_execution(
        mu=mu,
        alpha=alpha,
        m=m,
        p=p,
        ncores=ncores,
        store_result_edges=save_results,
        run_again=run_again, 
        path_data=path_data
    )
    logging.info(result)
