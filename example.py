import matplotlib.pyplot as plt
import networkx as nx
from sklearn.gaussian_process import kernels


from boostdags.causal_discovery import DAGBooster
from boostdags.pruning import gam_pruning
from simulation.graph import hamming_distance
from simulation.data import DataGenerator


p = 10
N = 100
seed = 21
kernel_func = kernels.RBF(1.0)


data_generator = DataGenerator(
    p, N,  kernel_func, seed=seed)

data_generator.generate_data()

kernel_func_estimator = kernels.RBF(1.0)

dag_boost = DAGBooster(kernel_func_estimator, data_generator.p)
dag_boost.train(data_generator.data)
# Optional pruning step
dag_boost_after_pruning = dag_boost.pruning(
    gam_pruning, data_generator.data)

data_generator.plot_graph()

adjacency_matrix_after_pruning = nx.adjacency_matrix(
    dag_boost_after_pruning, nodelist=range(p))
shd = hamming_distance(
    nx.adjacency_matrix(data_generator.graph), adjacency_matrix_after_pruning)

# Plot the graph after pruning
nx.draw_networkx(nx.from_scipy_sparse_array(
    adjacency_matrix_after_pruning, create_using=nx.DiGraph))
plt.show()
print(f"SHD: {shd}")
