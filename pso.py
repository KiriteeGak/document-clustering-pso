import numpy as np
from collections import defaultdict
import operator


class PSOInstance(object):
    def __init__(self, w2v_model, iterations, n_particles, n_clusters, inertia, local_accel_coeff, global_accel_coeff):
        self.ite = iterations
        self.doc_vecs = w2v_model / w2v_model.sum()
        self.n_particles = n_particles
        self.n_clusters = n_clusters
        self.inertia = inertia
        self.lac = local_accel_coeff
        self.gac = global_accel_coeff

    def generate_initial_velocities(self):
        velocities = np.random.uniform(-1, 1, (self.n_particles, self.n_clusters, np.shape(self.doc_vecs)[1]))
        return {swarm_index + 1: {cluster_index + 1: cluster_velocity for cluster_index, cluster_velocity in
                                  enumerate(particle_velocities)} for swarm_index, particle_velocities in
                enumerate(velocities)}

    def generate_initial_centroids(self):
        assert (self.n_clusters < np.shape(self.doc_vecs)[
            0]), "Number of clusters should be less than the number of data points"
        centroids = np.random.uniform(-1, 1, (self.n_particles, self.n_clusters, np.shape(self.doc_vecs)[1]))
        return {swarm_index + 1: {cluster_index + 1: clus_centroid for cluster_index, clus_centroid in
                                  enumerate(particle_centroid)} for swarm_index, particle_centroid in
                enumerate(centroids)}

    def assign_to_clusters(self, coordinates):
        swarm_assignment = defaultdict(dict)
        for swarm_index, cluster_centroids in coordinates.items():
            swarm_assignment[swarm_index] = self._minimum_distance(cluster_centroids, self.doc_vecs)
        return swarm_assignment

    def _minimum_distance(self, cluster, doc_vecs):
        doc_vec_assignment = defaultdict(list)
        doc_vecs_temp = doc_vecs[:]
        for id_, doc_vec in enumerate(doc_vecs):
            cluster_id = sorted([(cluster_id, np.linalg.norm(doc_vec-coord_)) for cluster_id, coord_ in cluster.items()], key=lambda k:k[1])[0][0]
            doc_vec_assignment[cluster_id].append(id_)
        return doc_vec_assignment

    def update_velocities(self, local_bests, global_bests, cog_factor, soc_factor, max_inertia, min_inertia):
        # vd = w * vd + c1 * rand * (p_id - x_id) + c2 * rand * (p_gd - x_id)  
        pass

    def update_centroids(self):
        pass

    def intra_cluster(self):
        pass

    def inter_cluster(self):
        pass

    def pso_main(self):
        for i in range(self.ite):
            if i == 0:
                (coordinates, velocities) = (self.generate_initial_centroids(), self.generate_initial_velocities())
            self.assign_to_clusters(coordinates)
            exit()


class LocalMinimum:
    def __init__(self, *args):
        self.local_best = {i: arg for i, arg in enumerate(args)}


class GlobalMinimum:
    def __init__(self, coordinates):
        self.global_best = coordinates


print PSOInstance(np.random.uniform(-2,2,(5,5)), 1, 4, 2, 0.01, 0.01, 0.01).main()
