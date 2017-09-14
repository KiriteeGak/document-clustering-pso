import numpy as np
from collections import defaultdict
from random import *

class BaseState:
    def __init__(self, clus_position=None, clus_velocity=None, clus_score=float('inf')):
        self.clus_position = clus_position
        self.clus_velocity = clus_velocity
        self.clus_score = clus_score

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
        for swarm_index, cluster_centroids in coordinates.iteritems():
            swarm_assignment[swarm_index] = self._minimum_distance(cluster_centroids, self.doc_vecs)
        return swarm_assignment

    def _minimum_distance(self, cluster, doc_vecs):
        doc_vec_assignment = defaultdict(list)
        doc_vecs_temp = doc_vecs[:]
        for id_, doc_vec in enumerate(doc_vecs):
            cluster_id = sorted([(cluster_id, np.linalg.norm(doc_vec-coord_)) for cluster_id, coord_ in cluster.iteritems()], key=lambda k:k[1])[0][0]
            doc_vec_assignment[cluster_id].append(id_)
        for clus_id in cluster: doc_vec_assignment[clus_id]
        return doc_vec_assignment

    def update_velocities(self, curr_velocities, local_bests, global_best, cog_factor, soc_factor, max_inertia, min_inertia):
        # v_id = w * v_id + c_1*r_1*(p_id-x_id) + c_2*r_2*(p_gd-x_id)
        _curr_velocities = curr_velocities.copy()
        for swarm_index, swarm_config in curr_velocities.iteritems():
            for clus_id, cluster_velocity in swarm_config.iteritems():
                local_best_vel = local_bests[swarm_index].clus_velocity[clus_id]
                _cluster_velocity = (max_inertia * cluster_velocity) + (cog_factor*random()*(local_best_vel - cluster_velocity)) + (soc_factor*random()*(global_best.clus_velocity[clus_id] - cluster_velocity))
                _curr_velocities[swarm_index][clus_id] = _cluster_velocity
        return _curr_velocities

    def update_centroids(self, evaluated_swarm, updated_velocities):
        _coordinates = defaultdict(dict)
        for swarm_index, swarm_config in evaluated_swarm.iteritems():
            for clus_id, position in swarm_config['configuration'].iteritems():
                _coordinates[swarm_index][clus_id] = position + updated_velocities[swarm_index][clus_id]
        return _coordinates

    def _swarm_evaluation(self, doc_vecs, association, cluster_centers, velocities):
        return {swarm_index : {'cluster_score' : self._davies_boudin_score(swarm_config, doc_vecs, cluster_centers[swarm_index]), 'configuration': cluster_centers[swarm_index], 'cluster_velocity' : velocities[swarm_index]}
         for swarm_index, swarm_config in association.iteritems()}

    def _davies_boudin_score(self, swarm_config, doc_vecs, cluster_centers):
        cluster_config_score = []
        for clus_id, docs in swarm_config.iteritems():
            if docs:
                _doc_vecs = [doc_vecs[doc_id] for doc_id in docs]
                cluster_config_score.append(sum(map(lambda a : np.linalg.norm(a-cluster_centers[clus_id]), _doc_vecs)))
            else:
                cluster_config_score.append(0)
        _score = 0
        for id_base, base_clus_config_score in enumerate(cluster_config_score):
            _score += max([((base_clus_config_score + iterative_clus_config_score)/float(np.linalg.norm(cluster_centers[id_base+1] - cluster_centers[id_iter+1]))) for id_iter, iterative_clus_config_score in enumerate(cluster_config_score) if id_base != id_iter])
        return _score/np.shape(doc_vecs)[0]

    def update_local_conf(self, evaluated_swarm, local_bests):
        local_bests_return = {}
        for swarm_index, state_instance in local_bests.iteritems():
            if evaluated_swarm[swarm_index]['cluster_score'] < state_instance.clus_score:
                local_bests_return[swarm_index] = BaseState(clus_position=evaluated_swarm[swarm_index]['configuration'], clus_velocity=evaluated_swarm[swarm_index]['cluster_velocity'], clus_score=evaluated_swarm[swarm_index]['cluster_score'])
            else:
                local_bests_return[swarm_index] = state_instance
        return local_bests_return

    def update_global_conf(self, evaluated_swarm, global_best):
        for swarm_index, swarm_config in evaluated_swarm.iteritems():
            if swarm_config['cluster_score'] <= global_best.clus_score:
                global_best = BaseState(clus_position=swarm_config['configuration'], clus_score=swarm_config['cluster_score'], clus_velocity=swarm_config['cluster_velocity'])
        return global_best

    def main(self):
        for i in range(self.ite):
            if i == 0:
                (coordinates, velocities) = (self.generate_initial_centroids(), self.generate_initial_velocities())
                local_bests = {swarm_index : BaseState(clus_position=cluster_config, clus_velocity=velocities[swarm_index]) for swarm_index, cluster_config in coordinates.iteritems()}
                global_best = choice(list(local_bests.values()))
            assignment = self.assign_to_clusters(coordinates)
            evaluated_swarm = self._swarm_evaluation(self.doc_vecs, assignment, coordinates, velocities)
            local_bests = self.update_local_conf(evaluated_swarm, local_bests)
            global_best = self.update_global_conf(evaluated_swarm, global_best)
            velocities = self.update_velocities(velocities, local_bests, global_best, 0.5, 0.5, 0.9, 0.5)
            coordinates = self.update_centroids(evaluated_swarm, velocities)
        return global_best.__dict__['clus_score']

print PSOInstance(np.random.uniform(-2,2,(10,10)), 100, 4, 2, 0.01, 0.01, 0.01).main()
# w2v_model, iterations, n_particles, n_clusters, inertia, local_accel_coeff, global_accel_coeff
# print PSOInstance(np.array([[1,1],[0,0],[1,0],[0,1],[10,10],[9,9],[10,9],[9,10]])-np.array([5,5]), 100, 5, 2, 0.5, 0.5, 0.5).main()