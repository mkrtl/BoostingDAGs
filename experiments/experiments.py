import logging
import multiprocessing as mp
import time

import networkx as nx
from notears import nonlinear
import numpy as np
from sklearn.gaussian_process import kernels

from boostdags.causal_discovery import DAGBooster
from boostdags.pruning import gam_pruning
from simulation.data import DataGenerator
from simulation.graph import hamming_distance

logging.basicConfig(level=logging.INFO)

root_path_results = "<PATH_TO_RESULTS>"
path_data = "<PATH_TO_DATA>"


def evaluate_graphs(estimated_graph, true_graph, p):
    estimated_adjacency_matrix = nx.adjacency_matrix(
        estimated_graph)  # , nodelist=range(p))
    true_adjacency_matrix = nx.adjacency_matrix(
        true_graph)  # , nodelist=range(p))

    shd = hamming_distance(
        true_adjacency_matrix, estimated_adjacency_matrix)
    logging.info("------------------------------------------")
    logging.info("Implemented SHD distance: " + str(shd))
    logging.info("------------------------------------------")
    n_true_positives = len([e for e in set(
        estimated_graph.edges) if e in true_graph.edges])

    n_false_positives = len([e for e in set(
        estimated_graph.edges) if e not in true_graph.edges])

    n_false_negatives = len([e for e in true_graph.edges if e not in set(
        estimated_graph.edges)])

    return dict(SHD=shd,
                n_true_positives=n_true_positives,
                n_false_positives=n_false_positives, n_false_negatives=n_false_negatives,
                n_edges_true_graph=len(true_graph.edges))


def run_experiment(seed=93,
                   mu=0.3,
                   alpha=.01,
                   m="AIC",
                   expected_n_edges=None,
                   p=100,
                   N=200,
                   save_graphs_and_result=False,
                   additive_structural_equations=True,
                   graph_type="ER",
                   n_max_parents=10,
                   ):
    kernel_func = kernels.RBF(1.0)
    kernel_func_estimator = kernels.RBF(1.)
    if not expected_n_edges:
        expected_n_edges = p

    save_path = f"{path_data}/dens_edges_{expected_n_edges:.2f}/"

    data_generator = DataGenerator(
        p, N,  kernel_func, seed=seed, graph_type=graph_type,
        expected_number_edges=expected_n_edges,
        structural_equation="additive" if additive_structural_equations else "non-additive")
    data_generator.generate_data()
    logging.info(f"Running experiment {seed}...")
    logging.info(
        f"The true graph has {len(data_generator.graph.edges)} edges.")
    adjacency_matrix_true = nx.adjacency_matrix(
        data_generator.graph, nodelist=range(p))

    dag_boost = DAGBooster(kernel_func_estimator, data_generator.p,
                           m=m, mu=mu, alpha=alpha, n_max_parents=n_max_parents)
    start = time.time()
    dag_boost.train(data_generator.data)
    dag_boost_after_pruning = dag_boost.pruning(
        gam_pruning, data_generator.data)
    end = time.time()
    duration = end - start
    adjacency_matrix_after_pruning = nx.adjacency_matrix(
        dag_boost_after_pruning, nodelist=range(p))
    shd = hamming_distance(
        adjacency_matrix_true, adjacency_matrix_after_pruning)

    n_true_positives = len([e for e in set(
        dag_boost_after_pruning.edges) if e in data_generator.graph.edges])

    n_false_positives = len([e for e in set(
        dag_boost_after_pruning.edges) if e not in data_generator.graph.edges])

    n_false_negatives = len([e for e in data_generator.graph.edges if e not in set(
        dag_boost_after_pruning.edges)])

    logging.info(
        f"The structural hamming distance is {shd}")
    if save_graphs_and_result:
        path_data = f"{save_path}/graph_{p}_{N}_additive_{additive_structural_equations}_{graph_type}/{seed}"
        data_generator.store_data_and_graph(path_data)

    return dict(SHD=shd, n_true_positives=n_true_positives, n_false_positives=n_false_positives, n_false_negatives=n_false_negatives,
                n_edges_true_graph=len(data_generator.graph.edges), seed=seed, duration=duration)


def main(mu=0.3,
         alpha=.01,
         n_exps=100,
         m="AIC",
         expected_n_edges=None,
         p=100,
         N=200,
         save_graphs_and_result=False,
         additive_structural_equations=True,
         graph_type="ER",
         n_max_parents=99,
         seed=93):
    main_seed = seed
    np.random.seed(main_seed)
    n_experiments = n_exps
    run_seeds = np.random.randint(0, 1e7, n_experiments)
    kernel_func = kernels.RBF(1.0)
    kernel_func_estimator = kernels.RBF(1.)

    if not expected_n_edges:
        expected_n_edges = 2/(p-1)
    logging.info(f"We run {n_experiments} experiments with {p} nodes, additive = {additive_structural_equations}, {N} samples, expected number of edges {expected_n_edges} and graph type {graph_type}.")
    save_path = f"{path_data}/dens_edges_{expected_n_edges:.2f}/"
    # results:
    shd_boosting = []
    shd_boosting_with_limited_edges = []
    shd_boosting_with_pns = []
    durations = []
    number_of_falsely_directed_edges = []
    number_of_falsely_directed_edges_with_limited_edges = []
    number_of_falsely_directed_edges_with_pns = []

    for seed in run_seeds[:n_exps]:
        data_generator = DataGenerator(
            p, N,  kernel_func, seed=seed, graph_type=graph_type,
            expected_number_edges=expected_n_edges,
            structural_equation="additive" if additive_structural_equations else "non-additive")
        data_generator.generate_data()
        logging.info(f"Running experiment {seed}...")
        logging.info(
            f"The true graph has {len(data_generator.graph.edges)} edges.")
        adjacency_matrix_true = nx.adjacency_matrix(
            data_generator.graph, nodelist=range(p))

        def run_dag_boost_variant_and_evaluate(type_of_estimate, pns_matrix=np.zeros((p, p), dtype=bool), n_edges_stop=(p * (p-1)) / 2):
            logging.info(f"Running {type_of_estimate}...")
            if type_of_estimate == "AIC":
                dag_boost = DAGBooster(kernel_func_estimator, data_generator.p,
                                       m=m, mu=mu, alpha=alpha, n_max_parents=n_max_parents)
                dag_boost.train(data_generator.data)
            elif type_of_estimate == "PNS":
                dag_boost = DAGBooster(kernel_func_estimator, data_generator.p, m=m,
                                       mu=mu, alpha=alpha,
                                       forbidden_edges=~pns_matrix, n_max_parents=n_max_parents)
                dag_boost.train(data_generator.data)
            elif type_of_estimate == "LIMITED_EDGE_NUMBER":
                dag_boost = DAGBooster(kernel_func_estimator, data_generator.p, m=10000,
                                       mu=mu, alpha=alpha, n_edges_stop=n_edges_stop, n_max_parents=n_max_parents)
                dag_boost.train(data_generator.data)
            adjacency_matrix_estimated = dag_boost.get_adjacency_matrix()
            logging.info(
                f"The estimator has {len(set(dag_boost.edges))} edges.")
            dag_boost_after_pruning = dag_boost.pruning(
                gam_pruning, data_generator.data)
            adjacency_matrix_after_pruning = nx.adjacency_matrix(
                dag_boost_after_pruning, nodelist=range(p))
            shd_before_pruning = hamming_distance(
                adjacency_matrix_true, adjacency_matrix_estimated)
            shd_after_pruning = hamming_distance(
                adjacency_matrix_true, adjacency_matrix_after_pruning)
            logging.info(
                f"The structural hamming distance is {shd_before_pruning}")
            logging.info(
                f"The structural hamming distance after pruning is {shd_after_pruning}")
            curr_number_of_falsely_directed_edges = len([e for e in set(
                dag_boost_after_pruning.edges) if (e[1], e[0]) in data_generator.graph.edges])
            logging.info(
                f"Number of edges with false direction {curr_number_of_falsely_directed_edges}")
            logging.info(f"The number of steps is {dag_boost.m_stop}")
            if type_of_estimate == "AIC":
                shd_boosting.append(shd_after_pruning)
                number_of_falsely_directed_edges.append(
                    curr_number_of_falsely_directed_edges)

            if type_of_estimate == "LIMITED_EDGE_NUMBER":
                shd_boosting_with_limited_edges.append(shd_after_pruning)
                number_of_falsely_directed_edges_with_limited_edges.append(
                    curr_number_of_falsely_directed_edges)
            if type_of_estimate == "PNS":
                shd_boosting_with_pns.append(shd_after_pruning)
                number_of_falsely_directed_edges_with_pns.append(
                    curr_number_of_falsely_directed_edges)

            logging.info("------------------------------------------")

        # DAGBoost
        start = time.time()
        run_dag_boost_variant_and_evaluate("AIC")
        end = time.time()
        durations.append(end - start)
        logging.info(f"Time for DAGBoost: {end-start}")
        if save_graphs_and_result:
            path_data = f"{save_path}/graph_{p}_{N}_additive_{additive_structural_equations}_{graph_type}/{seed}"
            data_generator.store_data_and_graph(path_data)

        print("------------------------------------------")

    return_dict = {"mean_falsely_directed_edges AIC": np.mean(number_of_falsely_directed_edges),
                   "sd falsely_directed_edges AIC": np.std(number_of_falsely_directed_edges),
                   "mean_shd AIC": np.mean(shd_boosting),
                   "sd_shd AIC": np.std(shd_boosting),
                   "mean_duration": np.mean(durations),
                   "std_duration": np.std(durations)}
    return return_dict


def main_notears(path):
    logging.info(f"Running experiment {path.split('/')[-1]}...")
    edges_path = "/edges.csv"
    data_path = "/data.csv"
    h_tol = 1e-4
    true_edges = np.loadtxt(f"{path}{edges_path}", delimiter=",", dtype="int")
    data = np.loadtxt(f"{path}/{data_path}", delimiter=",")
    p = data.shape[1]
    empty_graph = nx.empty_graph(p, create_using=nx.DiGraph)
    true_graph = nx.from_edgelist(true_edges, create_using=empty_graph)
    # Add nodes without adjacencies:
    nodes_to_add = [n for n in range(p) if n not in true_graph.nodes]
    true_graph.add_nodes_from(nodes_to_add)

    model_mlp = nonlinear.NotearsMLP(dims=[p, 10, 1])

    mlp_result = nonlinear.notears_nonlinear(
        model_mlp, data.astype("float32"), lambda1=0.03, lambda2=0.005, h_tol=h_tol)

    adjacaency_mlp = (np.abs(mlp_result) != 0.).astype(int)

    graph_mlp = nx.from_numpy_array(adjacaency_mlp, create_using=nx.DiGraph)

    evaluation_dict_mlp = evaluate_graphs(graph_mlp, true_graph, p)
    return dict(mlp=evaluation_dict_mlp)


def mu_sensitivity_analysis(mus, alpha=.01, n_exps=100, p=100, N=200, expected_n_edges=None, additive=True, graph_type="ER", m="AIC", n_max_parents=10):
    results = []
    for mu in mus:
        logging.info(f"mu: {mu}")
        result = main(mu=mu,
                      alpha=alpha,
                      n_exps=n_exps,
                      p=p,
                      N=N,
                      m=m,
                      expected_n_edges=expected_n_edges,
                      additive_structural_equations=additive,
                      graph_type=graph_type,
                      n_max_parents=n_max_parents)
        results.append(result)
    return results


def alpha_sensitivity_analysis(alphas, mu=.3, n_exps=100, p=100, N=200, expected_n_edges=None, additive=True, graph_type="ER", m="AIC", n_max_parents=10):
    results = []
    for alpha in alphas:
        logging.info(f"alpha: {alpha}")
        result = main(mu=mu,
                      alpha=alpha,
                      n_exps=n_exps,
                      p=p,
                      N=N,
                      m=m,
                      expected_n_edges=expected_n_edges,
                      additive_structural_equations=additive,
                      graph_type=graph_type,
                      n_max_parents=n_max_parents)
        results.append(result)
    return results


def parallel_notears_execution(paths, ncores=None):
    if ncores is None:
        ncores = mp.cpu_count()
    pool = mp.Pool(ncores)
    with pool:
        processes = [pool.apply_async(
            main_notears, args=(path,)) for path in paths]
        result = [p.get() for p in processes]
    return result


def parallel_experiment_execution(mu=0.3,
                                  alpha=.01,
                                  n_exps=100,
                                  m="AIC",
                                  expected_n_edges=None,
                                  p=100,
                                  N=200,
                                  save_graphs_and_result=False,
                                  additive_structural_equations=True,
                                  graph_type="ER",
                                  n_max_parents=99,
                                  main_seed=93,
                                  ncores=None):
    np.random.seed(main_seed)
    run_seeds = np.random.randint(0, 1e7, n_exps)

    if ncores is None:
        ncores = mp.cpu_count()
    pool = mp.Pool(ncores)
    with pool:
        processes = [pool.apply_async(run_experiment, kwds=dict(seed=seed, mu=mu, alpha=alpha, m=m,
                                                                expected_n_edges=expected_n_edges, p=p, N=N,
                                                                save_graphs_and_result=save_graphs_and_result,
                                                                additive_structural_equations=additive_structural_equations,
                                                                graph_type=graph_type, n_max_parents=n_max_parents)) for seed in run_seeds]
        result = [p.get() for p in processes]

    return result


if __name__ == "__main__":

    n_exps = 3
    p = 100
    N = 200
    expected_n_edges = p
    additive = True
    m = "AIC"
    graph_type = "ER"
    save_graphs_and_result = False
    n_max_parents = 10
    main_seed = 93
    ncores = 7
    mu = 0.3
    alpha = 0.1

    def target_path(
        t): return f"{root_path_results}/{t}_{N}_{p}_{expected_n_edges}_{graph_type}_{additive}_{n_exps}.csv"

    result = parallel_experiment_execution(mu=mu,
                                           alpha=alpha,
                                           n_exps=n_exps,
                                           m=m,
                                           expected_n_edges=expected_n_edges,
                                           p=p,
                                           N=N,
                                           save_graphs_and_result=save_graphs_and_result,
                                           additive_structural_equations=additive,
                                           graph_type=graph_type,
                                           n_max_parents=n_max_parents,
                                           main_seed=main_seed,
                                           ncores=ncores)
    logging.info(result)
    """
    target_path = f"{root_path_results}/{N}_{p}_{expected_n_edges}_{graph_type}_{additive}_{n_exps}_{main_seed}_{mu:.2f}_{alpha:.2f}.csv"
    pd.DataFrame(result).to_csv(target_path)

    result = main(mu=0.3,
                  alpha=.1,
                  n_exps=n_exps, m=m,
                  p=p, expected_n_edges=expected_n_edges,
                  save_graphs_and_result=save_graphs_and_result,
                  additive_structural_equations=additive,
                  graph_type=graph_type,
                  N=N,
                  notears=False,
                  n_max_parents=n_max_parents,
                  seed=93)
    mus = [0.3, 0.5, 0.7, 0.9]
    alphas = [0.001, 0.01, 0.1]

    def target_path(
        t): return f"{root_path_results}/mu_sensitivity/{t}_{N}_{p}_{expected_n_edges}_{graph_type}_{additive}_{n_exps}.csv"
    result = mu_sensitivity_analysis(mus,
                                     alpha=.01,
                                     n_exps=n_exps,
                                     m=m,
                                     p=p,
                                     expected_n_edges=expected_n_edges,
                                     additive=additive,
                                     graph_type=graph_type,
                                     N=N
                                     )
    logging.info(result)

    def target_path(
        t): return f"{root_path_results}/Sensitivity/{t}/{N}_{p}_{expected_n_edges}_{graph_type}_{additive}_{n_exps}.csv"
    pd.DataFrame(result).to_csv(target_path("alpha"))
    result2 = alpha_sensitivity_analysis(alphas,
                                         n_exps=n_exps,
                                         m=m,
                                         p=p,
                                         expected_n_edges=expected_n_edges,
                                         additive=additive,
                                         graph_type=graph_type,
                                         N=N
                                         )
    logging.info(result2)
    pd.DataFrame(result).to_csv(target_path("mu"))
    """
