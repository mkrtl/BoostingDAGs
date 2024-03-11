import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
import numpy as np
import seaborn as sns
import config

target_path = "./images/"
underfitting_file_name = "underfitting.png"
overfitting_file_name = "overfitting.png"
oracle_file_name = "oracle.png"

np.random.seed(12)
# sns.set_context("paper", rc=config.plot_context)
sns.set_context("talk")
N = 100
x_1 = np.random.normal(0, 1, N)
x_1 = x_1 - np.mean(x_1)
def f(x): return -3*np.cos(x)


rcParams.update(config.plot_context)

f_1 = f(x_1)
epsilon_2 = np.random.normal(0, .5**(1/2), N)
x_2 = f_1 + epsilon_2
mean_x_2 = np.mean(x_2)
x_2 = x_2 - mean_x_2
s = 50
x_1_linspace = np.linspace(min(x_1), max(x_1), 1000)
x_2_linspace = np.linspace(min(x_2), max(x_2), 1000)
figsize = (20, 10)

####################### Oracle ############################

### True ###
fig, ax = plt.subplots(figsize=figsize)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
sns.scatterplot(x=x_1, y=x_2, s=s)

sns.lineplot(x=x_1_linspace, y=f(x_1_linspace) -
             mean_x_2, linewidth=2, color="r")

plt.xlabel("$X_1$")
plt.ylabel("$X_2$")

fig.savefig(target_path + "true" + oracle_file_name, dpi=fig.dpi)

plt.show()


fig, ax = plt.subplots(figsize=figsize)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
sns.histplot(x=x_1, stat="density")

plt.xlabel("$X_1$")

fig.savefig(target_path + "hist_true" + oracle_file_name, dpi=fig.dpi)
print(
    f"Score Oracle True Ordering: {np.log(1/N * (np.sum(x_1**2))) + np.log(1/N * np.sum(epsilon_2**2))}")
plt.show()

### False ###
fig, ax = plt.subplots(figsize=figsize)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
sns.scatterplot(x=x_2, y=x_1, s=s)

sns.lineplot(x=x_2_linspace, y=np.zeros(1000), linewidth=2, color="r")

plt.xlabel("$X_2$")
plt.ylabel("$X_1$")
fig.savefig(target_path + "false" + oracle_file_name, dpi=fig.dpi)

plt.show()

fig, ax = plt.subplots(figsize=figsize)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
sns.histplot(x=x_2, stat="density")

plt.xlabel("$X_2$")

fig.savefig(target_path + "hist_false" + oracle_file_name, dpi=fig.dpi)
print(
    f"Score Oracle False Ordering: {np.log(1/N * np.sum(x_2**2)) + np.log(1/N * np.sum(x_1**2))}")
plt.show()

####################### Underfitting ############################
print("Underfitting")

# True Order
print("Correct order")
# Regression Plot
fig, ax = plt.subplots(figsize=figsize)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
sns.scatterplot(x=x_1, y=x_2, s=s)
sns.lineplot(x=x_1_linspace,
             y=np.mean(x_2) * np.ones(1000),
             linewidth=2,
             color="r")
plt.xlabel("$X_1$")
plt.ylabel("$X_2$")
fig.savefig(target_path + "true" + underfitting_file_name, dpi=fig.dpi)
plt.show()
# Histogram
fig, ax = plt.subplots(figsize=figsize)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
sns.histplot(x=x_1, stat="density")

plt.xlabel("$X_1$")

fig.savefig(target_path + "hist_true" + underfitting_file_name, dpi=fig.dpi)
print(
    f"Score Underfitting True Ordering: {np.log(1/N * np.sum(x_1**2)) + np.log(1/N * np.sum((x_2 - np.mean(x_2))**2))}")
plt.show()


print("Correct order")
# Regression Plot
fig, ax = plt.subplots(figsize=figsize)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
sns.scatterplot(x=x_2, y=x_1, s=s)
sns.lineplot(x=x_2_linspace,
             y=np.mean(x_1) * np.ones(1000),
             linewidth=2,
             color="r")
plt.xlabel("$X_2$")
plt.ylabel("$X_1$")
fig.savefig(target_path + "false" + underfitting_file_name, dpi=fig.dpi)
plt.show()
# Histogram
fig, ax = plt.subplots(figsize=figsize)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
sns.histplot(x=x_2, stat="density")

plt.xlabel("$X_2$")

fig.savefig(target_path + "hist_false" + underfitting_file_name, dpi=fig.dpi)
print(
    f"Score Underfitting False Ordering: {np.log(1/N * np.sum(x_2**2)) + np.log(1/N * np.sum((x_1 - np.mean(x_1))**2))}")
plt.show()

####################### Overfitting ############################
print("Overfitting")
print("True order")

fig, ax = plt.subplots(figsize=figsize)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
sns.scatterplot(x=x_1, y=x_2, s=s)

sns.lineplot(x=x_1,
             y=x_2,
             linewidth=2,
             color="r")
plt.xlabel("$X_1$")
plt.ylabel("$X_2$")

fig.savefig(target_path + "true_" + overfitting_file_name, dpi=fig.dpi)
plt.show()

print("False order")
fig, ax = plt.subplots(figsize=figsize)
sns.scatterplot(x=x_2, y=x_1, s=s)

sns.lineplot(x=x_2,
             y=x_1,
             linewidth=2,
             color="r")

plt.xlabel("$X_2$")
plt.ylabel("$X_1$")

fig.savefig(target_path + "false_" + overfitting_file_name, dpi=fig.dpi)
plt.show()
