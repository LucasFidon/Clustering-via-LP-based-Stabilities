"""
@brief  Implementation of "Clustering via LP-based Stabilities", N. Komodakis et al., NIPS 2009.

@author Lucas Fidon (lucas.fidon@kcl.ac.uk)
"""

import numpy as np


class Variables:
    def __init__(self, dist_matrix, mu=1.):
        dist_array = np.array(dist_matrix)
        # add penalization to avoid trivial solution
        C = mu*np.median(dist_array)  # heuristic choice for the penalty; default is the median of all distances
        penalization = C*np.ones((dist_array.shape[0]))
        dist_array += np.diag(penalization)
        self.d = np.copy(dist_array)  # distance matrix
        self.h = np.copy(dist_array)  # dual variables (can be interpreted as pseudo-distances)
        self.num_points = dist_array.shape[0]
        self.Q = []  # set of centroids id

    def margin(self, q):
        delta = 0.
        nn = np.argmin(self.h, axis=1)  # line-wise argmin
        # instead of computing the 2nd nearest neighbor,
        # computing the nearest neighbor if q is removed is enough (and simpler)
        nn_notq = np.argmin(np.delete(self.h, obj=q, axis=1), axis=1)  # line-wise argmin after removing column q
        nn_notq[nn_notq >= q] += 1
        for p in range(self.num_points):
            if not(p in self.Q):
                if self.h[p, q] == self.h[p, nn[p]]:  # q is the nearest neighbor of p
                    # stability of q as nn of p
                    delta += self.h[p, nn_notq[p]] - self.h[p, q]
                if p != q:
                    # regularization: penalize differences between h and d
                    delta -= self.h[p, q] - max(self.h[p, nn[p]], self.d[p, q])
        delta -= self.h[q, q] - self.h[q, nn[q]]
        return delta

    def test_dual_feasibility(self):
        """
        Raise an error if the current dual variables are not feasible for DUAL_Q
        :return: Null
        """
        # assert np.all(np.sum(self.h, axis=0) == np.sum(self.d, axis=0))
        for p in range(self.num_points):
            for q in range(self.num_points):
                if (p in self.Q) or (q in self.Q):
                    assert self.h[p, q] == self.d[p, q]
                elif p != q:
                    assert self.h[p, q] >= self.d[p, q]

    def dual_objective(self):
        dual_obj = np.sum(np.min(self.h, axis=1))
        return dual_obj

    def primal_objective(self):
        Q_c = [p for p in range(self.num_points) if not (p in self.Q)]  # complement of Q
        if len(self.Q) == 1:
            primal_obj = np.sum(self.d[:, self.Q[0]])
        else:
            primal_obj = np.sum(np.min(self.d[Q_c, :][:, self.Q], axis=1))
            primal_obj += np.sum(np.diag(self.d[self.Q, :][:, self.Q]))
        return primal_obj

    def project(self, q):
        # q is the new selected centroid
        for p in range(self.num_points):
            if not(p in self.Q):
                self.h[p, p] += self.h[q, p] - self.d[q, p]  # maintain feasibility using stack dual variables
                self.h[q, p] = self.d[q, p]
                self.h[p, q] = self.d[p, q]
        self.h[q, q] = self.d[q, q]  # not mentioned in the paper...

    def distribute(self):
        Q_c = [p for p in range(self.num_points) if not (p in self.Q)]  # complement of Q
        nn = np.argmin(self.h, axis=1)  # line-wise argmin
        h_min = np.min(self.h, axis=1)
        h_thres = np.copy(self.h)
        for p in Q_c:
            h_thres[p, nn[p]] = np.inf  # set minimum h_pq to infinity
        h_hat = np.min(h_thres, axis=1)  # next-to-minimum h_pq (h_hat_p)
        margins = [self.margin(q) for q in Q_c]
        L_Q = []  # list of objects whose min pseudo-distance is attained at an object from Q
        for p in Q_c:
            if nn[p] in self.Q:
                L_Q += [p]
        # update h_pq values
        for q in Q_c:
            margin_q = margins[Q_c.index(q)]
            V_q = [p for p in Q_c if (not(p in L_Q) and (h_min[p] >= self.d[p, q]))]
            if not(q in V_q):
                V_q += [q]
            card_V_q = len(V_q)
            for p in Q_c:
                if p != q and ((p in L_Q) or (h_min[p] < self.d[p, q])):
                    self.h[p, q] = max(h_min[p], self.d[p, q])
                elif self.h[p, q] > h_min[p]:
                    self.h[p, q] = h_min[p] - margin_q / card_V_q
                elif self.h[p, q] == h_min[p]:
                    self.h[p, q] = h_hat[p] - margin_q / card_V_q

    def search_for_new_stable_point(self):
        epsilon = 1e-5
        Q_c = [p for p in range(self.num_points) if not (p in self.Q)]  # complement of Q
        margins = [self.margin(q) for q in Q_c]
        dual_obj = self.dual_objective()
        max_margin = np.max(margins)
        print('  dual objective = %.2f, max margin = %.2f' % (dual_obj, max_margin))
        dual_obj_prev = np.inf
        while (max_margin < 0.) and (abs(dual_obj - dual_obj_prev)/self.num_points > epsilon):
            dual_obj_prev = dual_obj
            self.distribute()
            # distribute should maintain feasibility according to Theorem 3
            self.test_dual_feasibility()
            dual_obj = self.dual_objective()
            margins = [self.margin(q) for q in range(self.num_points) if not (q in self.Q)]
            max_margin = np.max(margins)
            print('  dual objective = %.2f, max margin = %.2f' % (dual_obj, max_margin))
        return Q_c[np.argmax(margins)]

    def clustering(self):
        self.test_dual_feasibility()
        new_candidate = self.search_for_new_stable_point()
        while self.margin(new_candidate) >= 0.:
            self.Q += [new_candidate]
            print('\nadd point %d to the centroids set (total: %d centroids)' % (new_candidate, len(self.Q)))
            # primal objective should strictly decrease wrt Theorem 4
            print('primal objective = %.2f' % self.primal_objective())
            self.project(new_candidate)
            self.test_dual_feasibility()
            new_candidate = self.search_for_new_stable_point()
