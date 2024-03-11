import numpy as np
from sklearn.gaussian_process import kernels

from simulation.data import DataGenerator
from boostdags.causal_discovery import SmallPDAGBooster
from simulation import graph

root_path_results = "<PATH_TO_RESULTS>"
path_data = "<PATH_TO_DATA>"

N = 200
additive = True
n_exps = 100


def main(mu=0.3,
         alpha=.1,
         n_exps=100,
         m="AIC",
         expected_n_edges=5,
         p=5,
         N=50,
         save_graphs_and_result=True,
         additive_structural_equations=True,
         graph_type="ER"):

    main_seed = 93
    np.random.seed(main_seed)
    n_experiments = n_exps
    run_seeds = np.random.randint(0, 1e7, n_experiments)
    kernel_func = kernels.RBF(1.)
    kernel_func_estimator = kernels.RBF(1.)
    if not expected_n_edges:
        expected_n_edges = p
    print(f"We run {n_experiments} experiments with {p} nodes, additive = {additive_structural_equations}, {N} samples, expected number of edges {expected_n_edges} and graph type {graph_type}.")
    save_path = f"{path_data}/small_p/dens_edges_{expected_n_edges:.2f}/"
    permutation_distances = []
    for seed in run_seeds[:n_exps]:
        print(f"Seed {seed}")
        data_generator = DataGenerator(
            p, N,  kernel_func, seed=seed, graph_type=graph_type,
            expected_number_edges=expected_n_edges,
            structural_equation="additive" if additive_structural_equations else "non-additive")
        data_generator.generate_data()
        dag_boost = SmallPDAGBooster(
            kernel_func_estimator, p, alpha=alpha, m=m, mu=mu)
        dag_boost.train(data_generator.data)

        permutation_distances.append(graph.min_transpositions_to_topo_ordering(
            data_generator.graph, dag_boost.best_permutation))
        if save_graphs_and_result:
            path_data = f"{save_path}/graph_{p}_{N}_additive_{additive_structural_equations}_{graph_type}/{seed}"
            data_generator.store_data_and_graph(path_data)
    return permutation_distances, run_seeds[:n_exps]


if __name__ == "__main__":

    distances, seeds = main(
        N=N, additive_structural_equations=additive, n_exps=n_exps)
    print(distances)
    print(seeds)
    print({k: v for (k, v) in zip(seeds, distances)})
    print(np.mean(distances))
    print(np.std(distances))
