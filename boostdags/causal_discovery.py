import itertools
import logging
from typing import Union

import networkx as nx
import numpy as np
from scipy import linalg


class BaseDAGBooster:

    def __init__(
        self,
        kernel_function: callable,
        p: int,
        m: Union[str, float, list[float], dict] = "AIC",
        mu: float = 0.3,
        alpha=0.01,
    ) -> None:
        """
        Base class that serves as a parent class for DAGBooster and SmallPDAGBooster.
        kernel_function: callable that takes two arguments and returns a scalar
        p: number of nodes
        m: number of boosting iterations or "AIC" for automatic stopping criterion
        mu: step size
        alpha: regularization parameter
        """
        # Maximal number of boosting iterations
        M_MAX = 10e3
        self.leave_out = False
        if m == "AIC":
            self.m = int(M_MAX)
            self.aic_scores_ = []
            self.aic_scores_individual = []
        elif isinstance(m, int):
            self.m = m
        elif isinstance(m, dict):
            self.leave_out_ratio = m["leave_out_ration"]
            self.leave_out = True
            self.m = int(M_MAX)
        else:
            raise Exception(
                f"Variable m must be either set to 'AIC' or an integer or a dict with key 'leave_out_ratio' and value between 0 and 1. You set it to {m}."
            )
        self.mu = mu
        self.p = p
        self.alpha = alpha
        self.kernel_function = kernel_function

        self.gram_matrices = None
        self.edges = list()
        self.aic = m == "AIC"

    def boosting_regression(
        self,
        X: np.ndarray,
        indices_of_parents: list[int],
        idx_target: int,
        m_stop: int = 100,
        mu: float = 0.3,
    ):
        """
        Regress idx_target on indices_of_parents.
        X: np.ndarray of shape (N, p)
        indices_of_parents: list of indices of parents
        idx_target: index of target node
        m_stop: number of boosting iterations
        mu: step size
        """
        u = X[:, idx_target].copy()
        chosen_parents = np.empty((m_stop,), dtype=np.int8)
        regression_estimate = np.zeros(X.shape[0])
        for m in range(m_stop):
            # Find regressor that minimizes the loss the most:
            scores_for_potential_parents = np.zeros(len(indices_of_parents))
            for k, parent in enumerate(indices_of_parents):
                scores_for_potential_parents[k] = np.linalg.norm(
                    self.calculate_kernel_regression(u, parent) - u, ord=2
                )
            idx_best_parent = indices_of_parents[
                np.argmin(scores_for_potential_parents)
            ]
            chosen_parents[m] = idx_best_parent
            regression_estimate += mu * self.calculate_kernel_regression(
                u, idx_best_parent
            )
            # Update u
            u = u - regression_estimate
        return chosen_parents

    def calculate_kernel_regression(self, u: np.ndarray, idx_parent: int) -> np.ndarray:
        hat_matrix = self.hat_matrices[idx_parent, :, :]
        u_hat = hat_matrix @ u
        return u_hat

    def calculate_gram_matrices(self, X: np.ndarray) -> np.ndarray:
        if self.gram_matrices is None:
            self.gram_matrices = np.zeros((self.p, X.shape[0], X.shape[0]))
        for idx_parent in range(X.shape[1]):
            self.gram_matrices[idx_parent, :, :] = (
                self.kernel_function(
                    X[:, idx_parent].reshape(-1, 1), X[:, idx_parent].reshape(-1, 1)
                )
                / X.shape[0]
            )
        return self.gram_matrices

    def get_graph(self) -> nx.DiGraph:
        nodes = list(range(self.p))
        graph = nx.DiGraph()
        graph.add_nodes_from(nodes)
        graph.add_edges_from(list(set(self.edges)))
        return graph

    def get_adjacency_matrix(self) -> np.ndarray:
        graph = self.get_graph()
        return nx.adjacency_matrix(graph, nodelist=range(self.p)).todense()

    def pruning(self, pruning_method, X: np.ndarray) -> list[tuple]:
        graph_before_pruning = self.get_graph()
        graph_after_pruning = nx.create_empty_copy(graph_before_pruning)
        for k in graph_before_pruning.nodes:
            logging.debug(f"Pruning parents of node {k}")
            possible_parents = list(graph_before_pruning.predecessors(k))
            logging.debug(f"The possible parents are {possible_parents}")
            chosen_parents_mask = pruning_method(X, possible_parents, k)
            chosen_parents = np.array(possible_parents)[chosen_parents_mask]
            logging.debug(f"The chosen parents are {chosen_parents}")
            edges_to_add = [(par, k) for par in chosen_parents]
            graph_after_pruning.add_edges_from(edges_to_add)
        return graph_after_pruning


class DAGBooster(BaseDAGBooster):

    def __init__(
        self,
        kernel_function: callable,
        p: int,
        m: Union[str, float, list[float], dict] = "AIC",
        mu: float = 0.3,
        alpha=0.01,
        forbidden_edges=None,
        n_edges_stop: int = None,
        n_max_parents: int = None,
    ) -> None:
        """
        Estimate DAG using boosting using a component-wise kernel regression.
        kernel_function: callable that takes two arguments and returns a scalar
        p: number of nodes
        m: number of boosting iterations, "AIC" for automatic stopping criterion or a dict with key 'leave_out_ratio' and value between 0 and 1 for leave-out validation
        mu: step size, between 0 and 1
        alpha: regularization parameter, positive
        forbidden_edges: np.ndarray of shape (p, p) with True for forbidden edges
        n_edges_stop: maximal number of edges in estimated graph. If None, then edge number is not restricted.
        n_max_parents: maximal number of parents for each node. If None, then number of parents is not restricted.
        """
        super().__init__(kernel_function, p, m=m, mu=mu, alpha=alpha)
        self.K_plus_lambda_inv = None
        self.scores = np.zeros((p, p))
        self.n_unique_edges_ = []
        self.m_stop = None
        self.sum_squared_residuals_ = []
        self.hat_matrices = None
        self.x_hats = None
        if forbidden_edges is None:
            self.forbidden_edges = np.zeros((p, p), dtype=bool)
        else:
            self.forbidden_edges = forbidden_edges
        self.edges_causing_cycle = np.eye(p, dtype=bool)
        self.path_matrix = np.eye(p, p)
        if n_edges_stop is None:
            self.n_edges_stop = p * (p - 1) / 2
        else:
            self.n_edges_stop = n_edges_stop
        if n_max_parents is None:
            self.n_max_parents = self.p - 1
        else:
            self.n_max_parents = n_max_parents
        self.betas = None

    def train(self, X: np.ndarray):
        # check if X has number of columns p
        assert X.shape[1] == self.p
        N = X.shape[0]
        if self.leave_out:
            n_leave_out = int(self.leave_out_ratio * X.shape[0])
            X = X[:-n_leave_out, :]
            X_test = X[-n_leave_out:, :]
            leave_out_residuals = np.sum(X_test**2, axis=0)
            fs_at_leave_out = np.zeros((n_leave_out, self.p))
            # self.betas contains the the regression coefficients for the functions of the N observations
            # for each potential parent (length p) and each potential target (length p)
            N = X.shape[0]
            self.betas = np.zeros((N, self.p, self.p))

       

        # calculate hat matrices
        self.calculate_hat_matrices(X)
        self.x_hats = np.zeros((self.p, self.p, N))
        self._resids_in_components = np.sum(X**2, axis=0)
        for k in range(self.p):
            self.calculate_score_improvement(X, k)
        us = X.copy()
        if self.aic:
            hat_matrices_multiplied = np.zeros_like(self.hat_matrices)
            for k in range(self.p):
                hat_matrices_multiplied[k, :, :] = np.eye(N)
            aic_score = np.inf
            self.scores_individual = np.zeros((self.p, 1))
        for m_curr in range(self.m):
            # get best edge
            idx_source, idx_target = self.get_best_edge()
            self.n_unique_edges_.append(len(set(self.edges)))
            # update forbidden edges
            self.update_edges_causing_cycle(idx_source, idx_target)
            # Update u of idx_target
            old_us_for_idx_target = us[:, idx_target].copy()
            delta_estimate_for_idx_target = self.mu * self.calculate_kernel_regression(
                us[:, idx_target], idx_source
            )
            if self.leave_out:
                # Update the betas defining the regression functions
                self.betas[:, idx_source, idx_target] += (
                    self.mu
                    * self.K_plus_lambda_inv[idx_source, :, :]
                    @ us[:, idx_target]
                )
                # The function values are now f_kj(x) = sum_ell(beta_(ell,j,k)^T K(x, x_(ell, j))
                f_at_leave_out_to_add = (
                    1
                    / n_leave_out
                    * (
                        self.betas[:, idx_source, idx_target]
                        @ self.kernel_function(
                            X_test[:, idx_source].reshape(-1, 1),
                            X[:, idx_source].reshape(-1, 1),
                        )
                    )
                )
                fs_at_leave_out[:, idx_target] += f_at_leave_out_to_add
                leave_out_residuals_node_old = leave_out_residuals[idx_target]
                new_leave_out_residuals_node = np.sum(
                    (X_test[:, idx_target] - fs_at_leave_out[:, idx_target]) ** 2
                )
                if new_leave_out_residuals_node > leave_out_residuals_node_old:
                    print(
                        f"The leave-out residual for node {idx_target} went up by {new_leave_out_residuals_node - leave_out_residuals_node_old}"
                    )
                    break
                leave_out_residuals[idx_target] = new_leave_out_residuals_node

            self.x_hats[idx_target, idx_source, :] += delta_estimate_for_idx_target
            us[:, idx_target] = old_us_for_idx_target - delta_estimate_for_idx_target
            # Update scores
            self._resids_in_components[idx_target] = np.sum(
                (np.sum(self.x_hats[idx_target, :, :], axis=0) - X[:, idx_target]) ** 2
            )  # np.sum((
            self.sum_squared_residuals_.append(np.sum(self._resids_in_components))
            for k in range(self.p):
                if k != idx_target:
                    # Update the forbidden edges
                    forbidden_parents = self.edges_causing_cycle[:, k] == 1
                    for j in range(self.p):
                        if forbidden_parents[j] or self.forbidden_edges[j, k]:
                            self.scores[j, k] = np.inf
                elif k == idx_target:
                    self.calculate_score_improvement(us, k)
            if self.aic:
                hat_matrices_multiplied[idx_target, :, :] = (
                    np.eye(N) - self.mu * self.hat_matrices[idx_source, :, :]
                ) @ hat_matrices_multiplied[idx_target, :, :]

                aic_score_old = aic_score
                traces_individual = N - np.trace(
                    hat_matrices_multiplied, axis1=1, axis2=2
                )
                sum_traces = np.sum(traces_individual)
                aic_score = np.sum(self._resids_in_components) + sum_traces
                aic_score_individual = self._resids_in_components + traces_individual
                self.aic_scores_individual.append(aic_score_individual)
                self.aic_scores_.append(aic_score)
                if aic_score_old < aic_score:
                    print(f"The AIC score went up by {aic_score - aic_score_old}")
                    break
                # If all possible remaining edges are already in graph, then stop.
                if (
                    len(
                        [
                            tuple(e)
                            for e in np.argwhere(self.scores != np.inf)
                            if (e[0], e[1]) not in self.edges
                        ]
                    )
                    == 0
                ):
                    break
            if len(set(self.edges)) >= self.n_edges_stop:
                break
            n_parents_for_idx_target = len(
                set([k for k, v in self.edges if v == idx_target])
            )
            if n_parents_for_idx_target >= self.n_max_parents:
                self.scores[:, idx_target] = np.inf
        self.m_stop = m_curr
        if self.aic:
            self.aic_scores_individual = np.concatenate(
                [np.reshape(sc, (1, -1)) for sc in self.aic_scores_individual], axis=0
            ).T
        return self.edges

    def calculate_K_plus_lambda_inv(self, X: np.ndarray) -> np.ndarray:
        if self.gram_matrices is None:
            self.calculate_gram_matrices(X)
        self.K_plus_lambda_inv = np.zeros((X.shape[1], X.shape[0], X.shape[0]))
        for idx in range(X.shape[1]):
            self.K_plus_lambda_inv[idx, :, :] = linalg.inv(
                self.gram_matrices[idx, :, :] + self.alpha * np.eye(X.shape[0])
            )
        return self.K_plus_lambda_inv

    def calculate_hat_matrices(self, X: np.ndarray) -> np.ndarray:
        hat_matrices = np.zeros((X.shape[1], X.shape[0], X.shape[0]))
        if self.gram_matrices is None:
            self.calculate_gram_matrices(X)
        if self.K_plus_lambda_inv is None:
            self.calculate_K_plus_lambda_inv(X)
        for idx in range(X.shape[1]):
            K_plus_lambda_inv = self.K_plus_lambda_inv[idx, :, :]
            hat_matrices[idx, :, :] = self.gram_matrices[idx, :, :] @ K_plus_lambda_inv
        self.hat_matrices = hat_matrices
        return hat_matrices

    def calculate_score_improvement(
        self, us: np.ndarray, idx_target: int
    ) -> np.ndarray:
        allowed_parents = (self.edges_causing_cycle[:, idx_target] == 0) & (
            self.forbidden_edges[:, idx_target] == False
        )
        forbidden_parents = (self.edges_causing_cycle[:, idx_target] == 1) | (
            self.forbidden_edges[:, idx_target] == True
        )
        idx_allowed_parents = np.where(allowed_parents)
        u_target = us[:, idx_target]
        u_hats = self.hat_matrices @ u_target

        for k in idx_allowed_parents[0]:
            self.scores[k, idx_target] = np.log(np.sum((u_target - u_hats[k]) ** 2))
        self.scores[:, idx_target] -= np.log(self._resids_in_components[idx_target])
        self.scores[forbidden_parents, idx_target] = np.inf
        return self.scores[:, idx_target]

    def get_best_edge(self) -> tuple[int]:
        # Taken from https://numpy.org/doc/stable/reference/generated/numpy.argmin.html
        best_edge = np.unravel_index(
            np.argmin(self.scores, axis=None), self.scores.shape
        )
        self.edges.append(best_edge)
        return best_edge

    def update_edges_causing_cycle(self, idx_source: int, idx_target: int) -> None:
        """
        Forbid the edges that would cause a cycle in the directed acyclic graph (DAG).
        """
        self.path_matrix[idx_source, idx_target] = 1
        # There is now a path from any ancestor of the source node to the target node.
        nodes_with_path_to_idx_source = np.where(self.path_matrix[:, idx_source] == 1)
        self.path_matrix[nodes_with_path_to_idx_source, idx_target] = 1
        # There is now a path from source node to any descendant of the target node.
        nodes_with_path_to_idx_target = np.where(self.path_matrix[idx_target, :] == 1)
        self.path_matrix[idx_source, nodes_with_path_to_idx_target] = 1
        self.edges_causing_cycle = self.path_matrix.T


class SmallPDAGBooster(BaseDAGBooster):
    """
    Estimate topological ordering of nodes. 
    Args: 
        kernel_function: callable that takes two arguments and returns a scalar
        p: number of nodes
        m: number of boosting iterations, "AIC" for automatic stopping criterion or a dict with key 'leave_out_ratio' and value between 0 and 1 for leave-out validation
        mu: step size, between 0 and 1
        alpha: regularization parameter, positive
    """

    def __init__(
        self,
        kernel_function: callable,
        p: int,
        m: Union[str, float, list[float]] = "AIC",
        mu: float = 0.3,
        alpha=0.01,
    ) -> None:
        super().__init__(kernel_function, p, m=m, mu=mu, alpha=alpha)

        self.best_permutation = None

    def train(self, X: np.ndarray):
        # Get all permutations of the nodes and return the permutation with the lowest score
        all_permutations = list(itertools.permutations(range(self.p)))
        scores = np.zeros(len(all_permutations))
        X_test = None
        if self.leave_out:
            n_leave_out = int(self.leave_out_ratio * X.shape[0])
            X = X[n_leave_out:, :]
            X_test = X[:, :n_leave_out]
        self.calculate_gram_matrices(X)
        for k, permutation in enumerate(all_permutations):
            scores[k] = self.train_permutation(X, permutation, X_test=X_test)

        idx_min = np.argmin(scores)
        best_permutation = all_permutations[idx_min]
        self.edges = [
            (best_permutation[k], best_permutation[k + 1]) for k in range(self.p - 1)
        ]
        self.best_permutation = best_permutation
        return self.edges

    def train_permutation(self, X: np.ndarray, permutation: list[int], X_test: np.ndarray = None) -> float:
        # check if X has number of columns p
        assert X.shape[1] == self.p
        N = X.shape[0]
        scores = np.zeros(self.p)

        for k, idx_target in enumerate(permutation):
            us = (X[:, idx_target] - np.mean(X[:, idx_target])).copy()
            if k == 0:
                scores[idx_target] = np.sum(us**2)
            else:
                indices_before_idx_target = permutation[:k]
                if self.aic:
                    aic_score = np.inf
                elif self.leave_out:
                    leave_out_residuals = np.sum(X[:, idx_target]**2, axis=0)
                    n_leave_out = X_test.shape[0]
                    fs_at_leave_out = np.zeros(n_leave_out)
                    # self.betas contains the the regression coefficients for the functions of the N observations
                    # for the parents and each target
                    self.betas = np.zeros(N)
                # The gram matrix is the sum of the gram matrices with index smaller than idx_target
                curr_gram_matrix = np.sum(
                    self.gram_matrices[indices_before_idx_target, :, :], axis=0
                )
                curr_K_plus_lambda_inv = linalg.inv(
                    curr_gram_matrix + self.alpha * np.eye(N)
                )
                # calculate hat matrix
                hat_matrix = curr_gram_matrix @ curr_K_plus_lambda_inv
                for m_curr in range(self.m):
                    us -= self.mu * hat_matrix @ us.copy()
                    if self.aic:
                        if m_curr == 0:
                            prod_identity_minus_hat_matrix = (
                                np.eye(N) - self.mu * hat_matrix
                            )
                        else:
                            prod_identity_minus_hat_matrix = (
                                prod_identity_minus_hat_matrix
                                @ (np.eye(N) - self.mu * hat_matrix)
                            )
                        trace_b_m = N - np.trace(prod_identity_minus_hat_matrix)
                        aic_score_new = np.sum(us**2) + trace_b_m
                        # For small N it might be that no convergence takes place.
                        if (m_curr > N) or (aic_score_new > aic_score):
                            break
                        else:
                            aic_score = aic_score_new
                    elif self.leave_out:
                        # Update the betas defining the regression functions
                        self.betas += (
                            self.mu * curr_K_plus_lambda_inv @ us
                        )
                        # The function values are now f_kj(x) = sum_ell(beta_(ell,j,k)^T K(x, x_(ell, j))
                        f_at_leave_out_to_add = (
                            1
                            / n_leave_out
                            * (
                                self.betas
                                @ self.kernel_function(
                                    X_test[:, indices_before_idx_target],
                                    X[:, indices_before_idx_target],
                                )
                            )
                        )
                        fs_at_leave_out += f_at_leave_out_to_add
                        leave_out_residuals_node_old = leave_out_residuals
                        new_leave_out_residuals_node = np.sum(
                            (X_test[:, idx_target] - fs_at_leave_out)
                            ** 2
                        )
                        if new_leave_out_residuals_node > leave_out_residuals_node_old:
                            break
                        leave_out_residuals = new_leave_out_residuals_node

                scores[idx_target] = np.sum(us**2)
        return np.sum(np.log(scores))
