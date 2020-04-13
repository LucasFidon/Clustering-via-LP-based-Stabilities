import numpy as np
import matplotlib.pyplot as plt
import os


def plot_clustering(x, y, centroids, centroids_init=None, title="My beautiful plot", filename="my_beautiful_plot"):
    """
    Plots data points whose color corresponds to their cluster and the centroids of each cluster
    :param x: Nx2 array; coordinates of the data points
    :param y: N array; cluster labels of the data points (from 0 to K-1, where K is the number of clusters)
    :param centroids: Kx2 array; coordinates of the centroids
    :param centroids_init: (optional) Kx2 array; coordinates of the initial centroids
    :param title: string; title for the figure that is saved on disk in the folder ./figs
    :param filename: string; file name for the figure that is saved on disk in the folder ./figs
    """
    print("Data and centroids are plotted...")
    K = centroids.shape[0]
    fig, ax = plt.subplots(figsize=(10, 10))
    # colors = ['blue', 'red', 'green', 'purple']
    color = iter(plt.cm.rainbow(np.linspace(0, 1, K)))
    for k in range(K):
        # Split data
        x_k = x[y == k, :]
        # Plot data of cluster y == k
        c = next(color)
        ax.scatter(x_k[:, 0], x_k[:, 1], c=c, marker='o', s=30, label='Cluster %d' % k)
    # Plot centroids
    ax.scatter(centroids[:, 0], centroids[:, 1], c='gold', marker='*', linewidth=2,
               s=500, label='Centroids')
    # (Optional) plot initial centroids (for K-means only)
    if centroids_init is not None:
        # Plot random centroids initialization
        ax.scatter(centroids_init[:, 0], centroids_init[:, 1], c='cyan',
                   marker='^', linewidth=2, s=220, label='Centroids init')
    # make the figure a bit more fancy
    ax.set_title(title, fontsize=24)
    xlim = [x[:, 0].min() - 0.5, x[:, 0].max() + 0.2]
    ax.set_xlim(xlim)
    ylim = [x[:, 1].min() - 0.2, x[:, 1].max() + 0.4]
    ax.set_ylim(ylim)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='upper left', fontsize=18, fancybox=True, shadow=True)
    # Save figure
    save_path = 'figs/%s.png' % filename
    if not(os.path.exists("figs")):
        os.makedirs("figs")
    nb = 0
    while os.path.exists(save_path):
        nb += 1
        save_path = 'figs/%s(%d).png' % (filename, nb)
    fig.savefig(save_path)
    print('The figure has been saved.')
