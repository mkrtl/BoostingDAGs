import logging
import random
import os

import numpy as np
import pandas as pd
import igraph
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


class DAGGenerator:

    def __init__(self, p: int, expected_number_edges: float = .3, seed=None) -> None:
        self.seed = seed
        np.random.seed(seed=self.seed)
        self.p = p
        self.expected_number_edges = expected_number_edges
        self.graph = nx.DiGraph()

    def generate_graph(self, graph_type="ER", seed=None) -> nx.Graph:
        """
        Generate a random graph with p nodes and density_edges edges.
        """
        if seed is None:
            seed = self.seed
        np.random.seed(self.seed)
        # Erdös-Renyi
        if graph_type == "ER":
            nodes = list(range(self.p))
            self.graph.add_nodes_from(nodes)

            np.random.shuffle(nodes)
            prob_edge = self.expected_number_edges * \
                2 / ((self.p - 1) * self.p)

            for order_target, k in enumerate(nodes):
                for j in nodes[:order_target]:
                    if (np.random.rand() < prob_edge):
                        self.graph.add_edge(
                            nodes[j], nodes[k])
        # Scale free graphs
        elif graph_type == "BA":
            # As in https://github.com/xunzheng/notears/blob/master/notears/utils.py#L17
            random.seed(self.seed.item())
            # graph = nx.scale_free_graph(self.p, seed=seed)
            graph = igraph.Graph.Barabasi(
                n=self.p, m=int(
                    self.expected_number_edges / self.p), directed=True).to_networkx()
            seeds_nodes = np.random.randint(0, 1e7, self.p)
            # Now permute node names as described here: https://stackoverflow.com/questions/59739750/how-can-i-randomly-permute-the-nodes-of-a-graph-with-python-in-networkx
            node_mapping = dict(zip(graph.nodes(), sorted(
                graph.nodes(), key=lambda k: np.random.uniform(seeds_nodes[k]))))
            self.graph = nx.relabel_nodes(graph, node_mapping)
        assert nx.is_directed_acyclic_graph(self.graph)
        return self.graph

    def plot_graph(self) -> None:
        """
        Plot the given graph.
        """
        nx.draw_networkx(self.graph)
        plt.show()


def hamming_distance(adjacency_1: np.ndarray, adjacency_2: np.ndarray) -> int:
    """
    https://github.com/ElementAI/causal_discovery_toolbox/blob/master/cdt/metrics.py
    """
    diff = np.abs(adjacency_1 - adjacency_2)
    diff = diff + diff.transpose()
    return int(np.sum(diff) / 2)


def transposition_distance(perm1: np.ndarray, perm2: np.ndarray):
    dist = 0
    for k, i in enumerate(perm1):
        for j in perm1[:k]:
            if perm2.index(j) > perm2.index(i):
                dist += 1
    return dist


def min_transpositions_to_topo_ordering(G, perm):
    topo_orderings = list(nx.all_topological_sorts(G))
    min_dist = float('inf')
    for topo_ordering in topo_orderings:
        dist = transposition_distance(topo_ordering, perm)
        if dist < min_dist:
            min_dist = dist
    return min_dist


def evaluate_graphs(estimated_graph, true_graph):
    estimated_adjacency_matrix = nx.adjacency_matrix(
        estimated_graph
    )  # , nodelist=range(p))
    true_adjacency_matrix = nx.adjacency_matrix(true_graph)  # , nodelist=range(p))
    shd = hamming_distance(true_adjacency_matrix, estimated_adjacency_matrix)
    logging.info("Implemented SHD distance: " + str(shd))
    
    logging.info("------------------------------------------")
    n_true_positives = len(
        [e for e in set(estimated_graph.edges) if e in true_graph.edges]
    )

    n_false_positives = len(
        [e for e in set(estimated_graph.edges) if e not in true_graph.edges]
    )

    n_false_negatives = len(
        [e for e in true_graph.edges if e not in set(estimated_graph.edges)]
    )

    return dict(
        SHD=shd,
        n_true_positives=n_true_positives,
        n_false_positives=n_false_positives,
        n_false_negatives=n_false_negatives,
        n_edges_true_graph=len(true_graph.edges),
    )


def write_edges_to_csv(graph, path, file_name="edges.csv"):
    if not os.path.exists(path):
            os.makedirs(path)
    path_edges = f"{path}/{file_name}"
    nx.write_edgelist(
            graph, path_edges, delimiter=",", data=False
        )
    
def load_data_and_graph(path_data):
    data = pd.read_csv(f"{path_data}/data.csv", header=None).to_numpy()
    edges = pd.read_csv(f"{path_data}/edges.csv", header=None)
    p = data.shape[1]
    true_graph = nx.DiGraph()
    true_graph.add_nodes_from(range(p))
    true_graph.add_edges_from(edges.values)
    return data, true_graph