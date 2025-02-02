import os

import networkx as nx
from simulation.graph import DAGGenerator
from simulation.gp import GPGenerator
from typing import Callable, Union
import numpy as np
from matplotlib import pyplot as plt


class DataGenerator(DAGGenerator):

    def __init__(
        self,
        p: int,
        N: int,
        kernel_func: Union[Callable, list[Callable]],
        expected_number_edges: int = None,
        graph_type: str = "ER",
        structural_equation="additive",
        scale: bool = False,
        noise_sd_with_parent: Union[float, list[float]] = None,
        noise_sd_without_parent: Union[float, list[float]] = None,
        N_for_normalization: int = 1000,
        seed: int = None,
    ) -> None:
        if expected_number_edges is None:
            expected_number_edges = p

        super().__init__(p, expected_number_edges=expected_number_edges, seed=seed)
        self._set_seeds(seed)
        self.scale = scale
        # Default values as in Causal Additive Models, Bühlmann et al.
        if noise_sd_with_parent is None:
            self.noise_sd = np.random.uniform(1 / 5, np.sqrt(2) / 5, p)
        elif isinstance(noise_sd_with_parent, float):
            self.noise_sd = [noise_sd_with_parent] * p
        if noise_sd_without_parent is None:
            self.noise_sd_without_parent = np.random.uniform(1, np.sqrt(2), p)
        elif isinstance(noise_sd_without_parent, float):
            self.noise_sd_without_parent = [noise_sd_without_parent] * p
        self.kernel_func = kernel_func
        if not isinstance(self.kernel_func, list) and not isinstance(
            self.kernel_func, Callable
        ):
            raise Exception("Parameter kernel_func must either be a list or a callable")

        self.graph = self.generate_graph(graph_type=graph_type)
        self.N = N
        """
        If scale is True, then we generate data for N + N_for_normalization samples.
        However, in the end we only return the data for N samples.
        """
        self.N_for_normalization = N_for_normalization
        if self.scale: 
            self.data = np.zeros((N + N_for_normalization, p))
            self.data_without_noise = np.zeros((N + N_for_normalization, p))
        else: 
            self.data = np.zeros((N, p))
            self.data_without_noise = np.zeros((N, p))
        
        self.structural_equation = structural_equation

    def _set_seeds(self, seed: int):
        self.seed = seed
        # np.random.seed(seed)
        self.seeds_noise = np.random.randint(0, 1e7, self.p)
        self.seeds_structural_equations = np.random.randint(0, 1e7, self.p)

    def generate_data(self, scale=None):
        """
        Generate data from the Structural Causal Model with the given graph and randomly sampled regression functions.

        If scale is True, data is does not fulfill varsortability and the signal-to-noise ratio is almost constant for each node.
        Further, the data is scaled to have mean 0 and standard deviation 1.
        """
        if scale is None:
            scale = self.scale
        if scale: 
            N_return_plus_normalization_set = self.N + self.N_for_normalization
        nodes_ordered = list(nx.topological_sort(self.graph))
        """
        If scale is True, then we generate data for N + N_for_normalization samples.
        However, in the end we only return the data for N samples.
        """
        for node in nodes_ordered:
            parents = list(self.graph.predecessors(node))
            # Calculate f_k(pa(k))
            self.data_without_noise[:, node] = (
                self._generate_structural_equations_of_submodel(parents, node)
            )
            # Add noise, so that X_k = f_k(pa(k)) + epsilon_k
            np.random.seed(self.seeds_noise[node])
            noise = np.random.normal(scale=self.noise_sd[node], size=N_return_plus_normalization_set)
            if scale and len(parents) > 0:
                # Scale the functions and use the the generated data of the remaining N_for_normalization samples
                # to estimate the mean and standard deviation of the data
                self.data_without_noise[:, node] -= np.mean(
                    self.data_without_noise[self.N:, node]
                )
                self.data_without_noise[:, node] /= np.std(
                    self.data_without_noise[self.N:, node]
                )
            # Now the noise-to-signal ratio (SNR) is exclusively determined by the noise
            # The SNR is 1 / (1 + self.noise_sd[node]^2)
            # Add noise, so that X_k = f_k(pa(k)) + epsilon_k
            self.data[:, node] = self.data_without_noise[:, node] + noise  
            if scale:
                # Scale the data using again the non-used samples
                self.data[:, node] -= np.mean(self.data[self.N:, node])
                self.data[:, node] /= np.std(self.data[self.N:, node]) 
        # Finally return only the data for the N samples
        self.data = self.data[: self.N, :]
        return self.data

    def _generate_structural_equations_of_submodel(
        self, parents: list[int], target_node: int
    ):
        # If there are no parents then use self.noise_sd_without_parent to increase variability in data.
        target_col = np.zeros(self.data.shape[0])
        if len(parents) == 0:
            if self.scale: 
                return target_col
            else: 
                np.random.seed(self.seeds_noise[target_node])
                return np.random.normal(
                    scale=self.noise_sd_without_parent[target_node],
                    size=self.N,
                )
                
        elif self.structural_equation == "additive":
            if isinstance(self.kernel_func, Callable):
                self.kernel_func = [self.kernel_func] * self.p

            gp_generators = [
                GPGenerator(
                    kernel_func_,
                    seed=self.seeds_structural_equations[target_node],
                    structure=self.structural_equation,
                )
                for kernel_func_ in self.kernel_func
            ]
            for parent in parents:
                # gp_generator = GPGenerator(self.kernel_func_, seed=seed)
                # Generate functions for every component
                target_col += gp_generators[parent].generate_noisefree_sample(
                    self.data[:, parent]
                )
        # Non-additive functions
        else:
            gp_generator = GPGenerator(
                self.kernel_func,
                seed=self.seeds_structural_equations[target_node],
                structure=self.structural_equation,
            )
            target_col += gp_generator.generate_noisefree_sample(self.data[:, parents])
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
        if self.scale: 
            data = self.data[: self.N, :]
        else: 
            data = self.data
        nx.write_edgelist(self.graph, path + "/edges.csv", delimiter=",", data=False)
        np.savetxt(path + "/data.csv", data, delimiter=",", fmt="%10f")
