"""
@brief  Implementation of the K-means clustering method for comparison with
        the clustering via LP-based stabilities method.

@author Lucas Fidon (lucas.fidon@kcl.ac.uk)
"""

import numpy as np
import random as rand
import utils
from plots import plot_clustering
from run_clustering import get_data


def get_centroids(x, y):
    K = np.max(y) + 1  # number of clusters
    centroids = []
    centroids_index = []
    for k in range(K):
        # get points belonging to the cluster k
        x_k = x[y == k, :]
        mu_k = np.mean(x_k, axis=0)
        # the centroid is the closest point to the mean of the cluster
        dist_k = np.sum(np.abs(x_k - mu_k[np.newaxis, :]), axis=1)
        i_centroid = np.argmin(dist_k)
        centroids += [x_k[i_centroid, :]]
        # convert index of the kth centroid in x_k to its index in x
        centroids_index += [np.where(y==k)[0][i_centroid]]
    return np.array(centroids), np.array(centroids_index)


def kmeans(x, K, rand_seed=0, verbose=False):
    num_points = x.shape[0]
    dist_matrix = utils.l2_distance_matrix(x)
    if verbose:
        print("start K-means from random initialization...")
    # random initialization
    rand.seed(rand_seed)  # allow to choose the evaluation
    # random initialization of the centroids
    centroids_index = np.array([rand.randint(0, num_points-1) for k in range(K)])
    # make sure we got K different points
    while np.unique(centroids_index).shape[0] < K:
        centroids_index = np.array([rand.randint(0, num_points - 1) for k in range(K)])
    centroids = np.array([x[i, :] for i in centroids_index])
    centroids_init = np.copy(centroids)  # returns for plots
    centroids_old = np.zeros((K, 2))
    # current classification of the data points
    y = (-1)*np.ones(num_points).astype(np.int)
    # update classification and centroids until convergence to a local minima
    iter = 0
    while np.linalg.norm(centroids - centroids_old) > 0:
        # each iteration is divided int 2 steps;
        # the distorsion should decrease during each step.
        # change classification
        y = utils.get_classification(dist_matrix, centroids_index)
        if verbose:
            distortion = utils.get_distortion(x, y, centroids)
            print("iter %d - step 1 (update classification): distortion=%.3f" % (iter, distortion))
        # change centroids
        centroids_old = centroids[:, :]
        centroids, centroids_index = get_centroids(x, y)
        if verbose:
            distortion = utils.get_distortion(x, y, centroids)
            print("iter %d - step 2 (update centroids): distortion=%.3f" % (iter, distortion))
        iter += 1
    if verbose:
        print("K-means has converged.")
    return y, centroids, centroids_init, centroids_index


def main(data_path, rand_seed):
    x, _, _ = get_data(data_path)
    dist_matrix = utils.l2_distance_matrix(x)
    y, centroids, centroids_init, centroids_index = kmeans(x, K, rand_seed)
    plot_clustering(x, y, centroids, centroids_init=centroids_init,
              title="Clustering using K-means",
              filename="kmean")
    distortion = utils.get_distortion(x, y, centroids)
    dunn = utils.Dunn_index(dist_matrix, y)
    davies = utils.DaviesBouldin_index(dist_matrix, y, centroids_index)
    silhouette = utils.Sihouette_index(dist_matrix, y)
    print('distorsion= %.2f, Dunn=%.3f, Davies-Bouldin=%.3f, Silhouette=%.3f' %
          (distortion, dunn, davies, silhouette))
    return distortion, dunn, davies, silhouette


if __name__ == '__main__':
    K = 4  # number of clusters expected
    data_path = '../../Data/Gaussian.data'
    distortions = []
    dunns = []
    davies_index = []
    silhouettes = []
    for rand_seed in range(100):
        print('----------- rand_seed=%d -----------' % rand_seed)
        distortion, dunn, davies, silhouette = main(data_path, rand_seed)
        distortions += [distortion]
        dunns += [dunn]
        davies_index += [davies]
        silhouettes += [silhouette]
    print('\n')
    print('---mean values---')
    print('distorsion= %.2f, Dunn=%.3f, Davies-Bouldin=%.3f, Silhouette=%.3f' %
         (np.mean(distortions), np.mean(dunns), np.mean(davies_index), np.mean(silhouettes)))
    print('---Std values---')
    print('distorsion= %.2f, Dunn=%.3f, Davies-Bouldin=%.3f, Silhouette=%.3f' %
          (np.std(distortions), np.std(dunns), np.std(davies_index), np.std(silhouettes)))
