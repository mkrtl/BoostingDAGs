import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
import numpy as np
import seaborn as sns
import config


target_path = "./images/"
file_correct_ordering = "two_d_sem.png"
file_incorrect_ordering = "two_d_sem_incorrect_ordering.png"
np.random.seed(12)
sns.set_context("paper", rc=config.plot_context)
N = 500
x_1 = np.random.normal(0, 1, N)
def f(x): return 3*-np.cos(x)


rcParams.update(config.plot_context)
plt.tight_layout()

f_1 = f(x_1)
x_2 = f_1 + np.random.normal(0, 1, N)
s = 3
dpi = config.plot_context["figure.dpi"]

figsize = config.plot_context["figure.figsize"]
fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
sns.scatterplot(x=x_1, y=x_2, s=s)
sns.lineplot(x=np.linspace(min(x_1), max(x_1), 1000),
             y=f(np.linspace(min(x_1), max(x_1), 1000)),
             linewidth=0.3,
             color="r")
plt.xlabel("$X_1$")
plt.ylabel("$X_2$")

fig.savefig(target_path + file_correct_ordering,
            dpi=fig.dpi, bbox_inches='tight')
fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
sns.scatterplot(x=x_2, y=x_1, s=s)
sns.lineplot(x=np.linspace(min(x_2), max(x_2), 1000),
             y=np.zeros(1000),
             linewidth=0.3,
             color="r")

plt.xlabel("$X_2$")
plt.ylabel("$X_1$")

fig.savefig(target_path + file_incorrect_ordering,
            dpi=fig.dpi, bbox_inches='tight')
