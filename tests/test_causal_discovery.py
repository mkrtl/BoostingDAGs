import unittest

import numpy as np
from sklearn.gaussian_process import kernels

from boostdags import causal_discovery


class TestBoostDAG(unittest.TestCase):

    def _init_dagbooster(self):
        kernel_func = kernels.RBF(1.0)
        p = 3
        m = 10
        return causal_discovery.DAGBooster(kernel_func, p, m)

    def test_update_forbidden_edges(self):
        """
        Take p = 3 and assume that there is edge from 0 to 1
        """
        dag_booster = self._init_dagbooster()
        forbidden_edges = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 1]])
        dag_booster.edges_causing_cycle = forbidden_edges
        """
        Now add the edge from 1 to 2. Then all edges but 1 -> 2 should be forbidden:
        """
        dag_booster.update_edges_causing_cycle(1, 2)
        expected_result = np.array([[1, 1, 0], [1, 1, 1], [1, 1, 1]])
        self.assertTrue(
            ((dag_booster.edges_causing_cycle - expected_result) == 0).all())

    def test_get_best_edge(self):
        dag_booster = self._init_dagbooster()
        dag_booster.scores = np.array([[np.inf, -2., -4.],
                                       [-3., np.inf, np.inf],
                                       [-7.2, np.inf, np.inf]])

        best_edge = dag_booster.get_best_edge()
        expected_result = (2, 0)
        self.assertTupleEqual(best_edge, expected_result)

    def test_gram_matrices(self):
        pass


if __name__ == "__main__":
    unittest.main()
