"""
@brief  Implementation of three widely used internal cluster validity measures:
        the Dunn index (Dunn 1974),
        the Davies-Bouldin index (Davies 1979),
        and the Silhouette index (Rousseuw 1987).

@author Lucas Fidon (lucas.fidon@kcl.ac.uk)
"""

import numpy as np


def l2_distance_matrix(design_matrix):
    n = design_matrix.shape[0]
    dist_matrix = np.zeros((n, n))
    for i in range(1, n):
        for j in range(i):
            d_ij = np.linalg.norm(design_matrix[i, :] - design_matrix[j, :])
            dist_matrix[i, j] = d_ij
            dist_matrix[j, i] = d_ij
    return dist_matrix


def get_classification(dist_matrix, centroids_id):
    # return the id of the closest centroids from each point
    dist_to_centroids = dist_matrix[:, centroids_id]
    closest_centroid = np.argmin(dist_to_centroids, axis=1)
    return closest_centroid


def Dunn_index(dist_matrix, y):
    K = np.max(y) + 1  # number of clusters
    # distance between clusters
    def dist_ci_cj(ci, cj):
        assert ci != cj
        return np.min(dist_matrix[y==ci, :][:, y==cj])
    # diameter of a cluster
    def diam_c(c):
        return np.max(dist_matrix[y==c, :][:, y==c])
    max_diam = np.max([diam_c(c) for c in range(K)])
    min_dist_inter_cluster = np.min(
        [np.min([dist_ci_cj(ci, cj) for ci in range(cj)]) for cj in range(1,K)]
    )
    return min_dist_inter_cluster/max_diam


def DaviesBouldin_index(dist_matrix, y, centroid_index):
    K = np.max(y) + 1  # number of clusters
    # Similarity of points in the same cluster
    def similarity(c):
        return np.mean(dist_matrix[y==c, :][:, centroid_index[c]])
    # Dispersion between two cluster
    def dispersion(ci, cj):
        assert ci != cj
        return dist_matrix[centroid_index[ci], centroid_index[cj]]
    # Dissimilarity between two clusters
    def dissimilarity(ci, cj):
        return (similarity(ci) + similarity(cj))/dispersion(ci, cj)
    DB = 0.
    for k in range(K):
        DB += np.max([dissimilarity(i,k) for i in range(K) if i != k])
    DB /= K
    return DB


def Sihouette_index(dist_matrix, y):
    K = np.max(y) + 1  # number of clusters
    # Silhouette index of a cluster
    def silhouette_c(c):
        # average distance between sample p and all the remaining elements assigned to the same cluster
        a = np.mean(dist_matrix[y==c, :][:, y==c], axis=1)
        n_c = a.shape[0]  # number of points in cluster c
        # minimum average distance between each sample of the cluster c to other clusters
        b = np.min(
            np.stack(
                [np.mean(dist_matrix[y==c, :][:, y==c2], axis=1) for c2 in range(K) if c2 != c],
                axis=1),
            axis=1
        )
        s = (b - a) / np.maximum(a, b)
        return np.sum(s)/n_c
    # Global silhouette index
    GS = np.mean([silhouette_c(c) for c in range(K)])
    return GS


# Other evaluation criteria
def get_distortion(x, y, centroids):
    distortion = 0
    for k in range(centroids.shape[0]):
        # distortion of cluster k
        distortion += np.sum(np.square(x[y == k, :] - centroids[k, :]))
    return distortion
