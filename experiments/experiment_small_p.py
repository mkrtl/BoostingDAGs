import os

import numpy as np
from sklearn.gaussian_process import kernels

from boostdags.causal_discovery import SmallPDAGBooster
from simulation import graph
import constants

graph_type = "ER"

additive = True
N = 50
p = 5
expected_n_edges = p
path_data = constants.PATH_DATA_SMALL_P(N, "scaled", p, additive, graph_type)
# Stopping criterion for the boosting algorithm
m = "AIC"

def main(
    path_data,
    mu=0.3,
    alpha=0.01,
    m="AIC",
    p=5,
):

    kernel_func_estimator = kernels.RBF(1.0)
    permutation_distances = []
    for k, path in enumerate(
        [
            os.path.join(os.path.abspath(path_data), p)
            for p in os.listdir(path_data)
            if os.path.isdir(os.path.join(path_data, p))
        ],
    ):
        data, true_graph = graph.load_data_and_graph(path)
        dag_boost = SmallPDAGBooster(kernel_func_estimator, p, alpha=alpha, m=m, mu=mu)
        dag_boost.train(data)
        permutation_distances.append(
            graph.min_transpositions_to_topo_ordering(
                true_graph, dag_boost.best_permutation
            )
        )
        print(f"Experiment {k}: {permutation_distances[-1]}")
    return permutation_distances


if __name__ == "__main__":

    distances = main(
        path_data,
        p=p,
        m=m
    )
    print(distances)
    print(np.mean(distances))
    print(np.std(distances))
