"""
This script generates data for the experiments in the paper. It generates data for a given number of experiments, where each experiment consists of a dataset and a corresponding graph. The data is generated using a Gaussian Process with a given kernel function. The graph is generated using a random graph model (Erdos-Renyi or Barabasi-Albert) with a given number of expected edges. The data is generated using a structural equation model with additive or non-additive noise. The data and the graph are stored in a given directory.
"""
import logging

import numpy as np
from sklearn.gaussian_process import kernels

from simulation.data import DataGenerator
import constants


main_seed = constants.MAIN_SEED

scale_data = True
additive_structural_equations = False
graph_type = "ER"

N = 10
p = 5
expected_n_edges = p
small_p = True
path_data = (constants.PATH_DATA_SMALL_P(N, "scaled", p, additive_structural_equations, graph_type) if small_p 
             else constants.PATH_DATA_LARGE_P("scaled", p, N, additive_structural_equations, graph_type))

def store_data_and_graph(
    seed,
    expected_n_edges=expected_n_edges,
    p=p,
    N=N,
    save_graphs_and_result=True,
    additive_structural_equations=True,
    graph_type="ER",
    path_data=path_data,
    scale_data=True,
):

    kernel_func = kernels.RBF(1.0)
    if not expected_n_edges:
        expected_n_edges = p

    save_path = (
        f"{path_data}/dens_edges_{expected_n_edges:.2f}/scaled"
        if scale_data
        else f"{path_data}/dens_edges_{expected_n_edges:.2f}/unscaled"
    )
    data_generator = DataGenerator(
        p,
        N,
        kernel_func,
        seed=seed,
        graph_type=graph_type,
        expected_number_edges=expected_n_edges,
        structural_equation=(
            "additive" if additive_structural_equations else "non-additive"
        ),
        scale=scale_data,
        N_for_normalization=200
    )
    data_generator.generate_data()
    logging.info(f"Generated data for experiment {seed}...")
    if save_graphs_and_result:
        path_data = f"{save_path}/graph_{p}_{N}_additive_{additive_structural_equations}_{graph_type}/{seed}"
        data_generator.store_data_and_graph(path_data)


def store_data_and_graphs(
    n_exps=100,
    p=p,
    N=N,
    save_graphs_and_result=True,
    additive_structural_equations=True,
    graph_type="ER",
    path_data=path_data,
    scale_data=True,
    main_seed=main_seed,
):

    np.random.seed(main_seed)
    run_seeds = np.random.randint(0, 1e7, n_exps)
    for seed in run_seeds[:n_exps]:
        store_data_and_graph(
            seed,
            p=p,
            N=N,
            save_graphs_and_result=save_graphs_and_result,
            additive_structural_equations=additive_structural_equations,
            graph_type=graph_type,
            path_data=path_data,
            scale_data=scale_data,
        )


if __name__ == "__main__":
    store_data_and_graphs(
        path_data=path_data,
        scale_data=scale_data,
        main_seed=main_seed,
        additive_structural_equations=additive_structural_equations,
        graph_type=graph_type,
    )
