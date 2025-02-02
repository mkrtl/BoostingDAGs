import logging
import multiprocessing as mp
import os

import networkx as nx
import numpy as np

from simulation.graph import evaluate_graphs, write_edges_to_csv, load_data_and_graph
import constants
from notears import nonlinear

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
target_file_name = "notears_edges.csv"
# Run experiment even if <target_file_name> already exists in target directory?
run_again = True


store_result_edges = True

path_data = constants.PATH_DATA_LARGE_P(scaled_string, p, N, additive, graph_type)


def main_notears(path, store_result_edges=True, target_file_name=target_file_name, run_again=run_again):
    logging.info(f"Running experiment {path.split('/')[-1]}...")
    
    h_tol = 1e-4
    
    data, true_graph = load_data_and_graph(path)
    p = data.shape[1]
    if os.path.exists(os.path.join(path, target_file_name)) and run_again is False:
        logging.info(f"Experiment {path} already exists.")
        return None
    model_mlp = nonlinear.NotearsMLP(dims=[p, 10, 1])

    mlp_result = nonlinear.notears_nonlinear(
        model_mlp, data.astype("float32"), lambda1=0.03, lambda2=0.005, h_tol=h_tol
    )

    adjacaency_mlp = (np.abs(mlp_result) != 0.0).astype(int)

    graph_mlp = nx.from_numpy_array(adjacaency_mlp, create_using=nx.DiGraph)
    evaluation_dict_mlp = evaluate_graphs(graph_mlp, true_graph)
    if store_result_edges:
        write_edges_to_csv(graph_mlp, path, file_name=target_file_name)
    return dict(mlp=evaluation_dict_mlp)


def parallel_notears_execution(path_data, ncores=None):
    if ncores is None:
        ncores = mp.cpu_count()
    pool = mp.Pool(ncores)
    paths_data = [
        os.path.join(os.path.abspath(path_data), p)
        for p in os.listdir(path_data)
        if os.path.isdir(os.path.join(path_data, p))
    ]
    if ncores is None:
        ncores = mp.cpu_count()
    pool = mp.Pool(ncores)
    with pool:
        processes = [pool.apply_async(main_notears, args=(path,)) for path in paths_data]
        result = [p.get() for p in processes]
    return result


if __name__ == "__main__":
    results = parallel_notears_execution(path_data)
    print(results)