import numpy as np
import utils
from clustering_via_lp_based_stabilitiesdpy import Variables
from plots import plot_clustering


def get_data(file_path, mean=None, std=None):
    """
    Load data and normalize them to zero mean and unit variance.
    :param file_path:
    :return: x, mean, std: x is a (design) matrix of shape (n_points, n_coordinates)
    """
    with open(file_path, 'r') as f:
        data = f.readlines()
    # One point per line
    # line format: "x_coord y_coord\n"
    data = [x.replace('\n', '').split(' ') for x in data]
    x = np.array(data).astype(np.float64)
    # Normalize data to zero mean and unit variance
    if mean is None:
        mean = np.mean(x, axis=0)
    x -= mean[np.newaxis,:]
    if std is None:
        std = np.std(x, axis=0)
    x /= std[np.newaxis, :]
    return x, mean, std


def main(data_path):
    design_matrix, _, _ = get_data(data_path)
    dist_matrix = utils.l2_distance_matrix(design_matrix)
    # apply the algorithm
    mu = 30.
    var = Variables(dist_matrix, mu=mu)
    var.clustering()
    centroids_id = var.Q
    y = utils.get_classification(dist_matrix, centroids_id)
    centroids = design_matrix[centroids_id, :]
    # plot result
    plot_clustering(design_matrix, y, centroids,
              title="Clustering via LP-based Stabilities (C=%.1f)" % mu,
              filename="lp_stabilities")
    distortion = utils.get_distortion(design_matrix, y, centroids)
    dunn = utils.Dunn_index(dist_matrix, y)
    davies = utils.DaviesBouldin_index(dist_matrix, y, centroids_id)
    silhouette = utils.Sihouette_index(dist_matrix, y)
    print('distorsion= %.2f, Dunn=%.3f, Davies-Bouldin=%.3f, Silhouette=%.3f' %
          (distortion, dunn, davies, silhouette))


if __name__ == '__main__':
    data_path = '../../Data/Gaussian.data'
    main(data_path)
