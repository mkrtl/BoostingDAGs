import os

import networkx as nx
from simulation.graph import DAGGenerator
from simulation.gp import GPGenerator
from typing import Callable, Union
import numpy as np
from matplotlib import pyplot as plt


class DataGenerator(DAGGenerator):

    def __init__(self, p: int, N: int, kernel_func: Union[Callable, list[Callable]], expected_number_edges: int = None,
                 graph_type: str = "ER",
                 structural_equation="additive",
                 noise_variance_with_parent: Union[float, list[float]] = None,
                 noise_variance_without_parent: Union[float,
                                                      list[float]] = None,
                 seed: int = None) -> None:
        if expected_number_edges is None:
            expected_number_edges = p

        super().__init__(p, expected_number_edges=expected_number_edges, seed=seed)
        self._set_seeds(seed)

        # Default values as in Causal Additive Models, Bühlmann et al.
        if noise_variance_with_parent is None:
            self.noise_variance = np.random.uniform(1/5, np.sqrt(2)/5, p)
        elif isinstance(noise_variance_with_parent, float):
            self.noise_variance = [noise_variance_with_parent] * p
        if noise_variance_without_parent is None:
            self.noise_variance_without_parent = np.random.uniform(
                1, np.sqrt(2), p)
        elif isinstance(noise_variance_without_parent, float):
            self.noise_variance_without_parent = [
                noise_variance_without_parent] * p
        self.kernel_func = kernel_func
        if not isinstance(self.kernel_func, list) and not isinstance(self.kernel_func, Callable):
            raise Exception(
                "Parameter kernel_func must either be a list or a callable")

        self.graph = self.generate_graph(graph_type=graph_type)
        self.N = N
        self.data = np.zeros((N, p))
        self.structural_equation = structural_equation

    def _set_seeds(self, seed: int):
        self.seed = seed
        # np.random.seed(seed)
        self.seeds_noise = np.random.randint(0, 1e7, self.p)
        self.seeds_structural_equations = np.random.randint(0, 1e7, self.p)

    def generate_data(self):
        """
        Generate data from the Structural Causal Model with the given graph and randomly sampled regression functions.
        """
        nodes_ordered = list(nx.topological_sort(self.graph))
        for node in nodes_ordered:
            parents = list(self.graph.predecessors(node))
            # Calculate f_k(pa(k))
            self.data[:, node] = self._generate_structural_equations_of_submodel(
                parents, node)
            # Add noise, so that X_k = f_k(pa(k)) + epsilon_k
            np.random.seed(self.seeds_noise[node])
            self.data[:, node] += np.random.normal(
                scale=self.noise_variance[node], size=self.N)
        return self.data

    def _generate_structural_equations_of_submodel(self, parents: list[int], target_node: int):
        # If there are no parents then use self.noise_variance_without_parent to increase variability in data.
        target_col = np.zeros(self.data.shape[0])
        if len(parents) == 0:
            np.random.seed(self.seeds_noise[target_node])
            return np.random.normal(
                scale=self.noise_variance_without_parent[target_node],
                size=self.N,
            )
        elif self.structural_equation == "additive":
            if isinstance(self.kernel_func, Callable):
                self.kernel_func = [self.kernel_func] * self.p

            gp_generators = [GPGenerator(kernel_func_,
                                         seed=self.seeds_structural_equations[target_node],
                                         structure=self.structural_equation)
                             for kernel_func_ in self.kernel_func]
            for parent in parents:
                # gp_generator = GPGenerator(self.kernel_func_, seed=seed)
                # Generate functions for every component
                target_col += gp_generators[parent].generate_noisefree_sample(
                    self.data[:, parent])
        # Non-additive functions
        else:
            gp_generator = GPGenerator(self.kernel_func,
                                       seed=self.seeds_structural_equations[target_node],
                                       structure=self.structural_equation)
            target_col += gp_generator.generate_noisefree_sample(
                self.data[:, parents])
        return target_col

    def plot_data(self, source_node: int, target_node: int):
        plt.scatter(self.data[:, source_node], self.data[:, target_node])
        plt.show()

    def store_data_and_graph(self, path: str):
        """
        Store data and graph in a csv file.
        """
        if not os.path.exists(path):
            os.makedirs(path)
        nx.write_edgelist(self.graph, path + "/edges.csv",
                          delimiter=",", data=False)
        np.savetxt(path + "/data.csv", self.data, delimiter=",", fmt='%10f')
