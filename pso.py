import numpy as np
from collections import defaultdict
from random import *
from cluster_evaluation import *


class BaseState:
    """BaseState depicts the properties of each cluster. All swarms down the node are pointed to BaseClass"""

    def __init__(self, cluster_position=None, cluster_velocity=None, cluster_score=float('inf'), association=None):
        self.cluster_position = cluster_position
        self.cluster_velocity = cluster_velocity
        self.cluster_score = cluster_score
        self.association = association


class PSOInstance(object):
    """Creates and runs the pso model with all params pointed as class args"""

    def __init__(self, w2v_model, iterations, n_particles, n_clusters, inertia, local_accel_coeff, global_accel_coeff):
        """
        :param w2v_model: A Numpy nd-array of word vectors obtained from doc2vec
        :param iterations: The number of iterations particle system should undergo
        :param n_particles: The number of swarm solutions to be created
        :param n_clusters: Number of clusters to be formed in each swarm
        :param inertia: Constant ability of particle to consider its previous velocity
        :param local_accel_coeff: Weight given to the local best velocity 
        :param global_accel_coeff: Weight given to the global best velocity
        """
        self.ite = iterations
        self.doc_vecs = w2v_model / np.fabs(w2v_model).max()
        self.n_particles = n_particles
        self.n_clusters = n_clusters
        self.inertia = inertia
        self.lac = local_accel_coeff
        self.gac = global_accel_coeff


    def _clamp_boundaries(self, doc_vecs):
        return [[min(doc_vecs[:,col_id]), max(doc_vecs[:,col_id])] for col_id in range(doc_vecs.shape[1])]

    def _clamp_velocities(self, clamped_boundaries):
        pass

    def generate_initial_velocities(self):
        """ Randomly initialise velocity vectors for each cluster in each swarm
        :return: Random velocities for each cluster from each swarm
        """
        velocities = np.random.uniform(-1, 1, (self.n_particles, self.n_clusters, np.shape(self.doc_vecs)[1]))
        return {swarm_index + 1: {cluster_index + 1: cluster_velocity for cluster_index, cluster_velocity in
                                  enumerate(particle_velocities)} for swarm_index, particle_velocities in
                enumerate(velocities)}

    def generate_initial_centroids(self):
        """ Randomly initialise coordinate vectors for each cluster in each swarm
        :return: Random coordinates generated for each cluster from each swarm
        """
        assert (self.n_clusters <= np.shape(self.doc_vecs)[
            0]), "Number of clusters should be less than the number of data points"
        centroids = np.random.uniform(-1, 1, (self.n_particles, self.n_clusters, np.shape(self.doc_vecs)[1]))
        return {swarm_index + 1: {cluster_index + 1: cluster_centroid for cluster_index, cluster_centroid in
                                  enumerate(particle_centroid)} for swarm_index, particle_centroid in
                enumerate(centroids)}

    def assign_to_clusters(self, coordinates):
        """ Assign points to cluster coordinates for each swarm
        :param coordinates: The current coordinates of each cluster
        :return: The vectors indices assigned to each cluster for each swarm
        """
        swarm_assignment = defaultdict(dict)
        for swarm_index, cluster_centroids in coordinates.iteritems():
            swarm_assignment[swarm_index] = self._minimum_distance(cluster_centroids)
        return swarm_assignment

    def _minimum_distance(self, cluster):
        """ Returns the map of cluster id to list of documents attached
        :param cluster: The coordinates for each cluster.
        :return: Cluster id and the corresponding documents assigned to it.
        """
        doc_vec_assignment = defaultdict(list)
        for id_, doc_vec in enumerate(self.doc_vecs):
            cluster_id = \
             sorted([(cluster_id, np.linalg.norm(doc_vec - coord_)) for cluster_id, coord_ in cluster.iteritems()],
                    key=lambda k: k[1])[0][0]
            doc_vec_assignment[cluster_id].append(id_)
        for cluster_id in cluster:
            if cluster_id not in doc_vec_assignment:
                doc_vec_assignment[cluster_id] = list()
        return doc_vec_assignment

    @staticmethod
    def _update_velocities(curr_velocities, evaluated_swarm, local_bests, global_best, cognitive_factor, social_factor,
                           max_inertia, min_inertia, implementation='pswv'):
        """The updated velocities based on cognitive, social factors and local/global minimum
        :param curr_velocities: The current velocities of each cluster in each swarm
        :param local_bests: The best configuration encountered in each index for each cluster
        :param global_best: The best global swarm config encountered in the swarm
        :param cognitive_factor: The importance given to the historical local best position
        :param social_factor: The importance given to the global best position
        :param max_inertia: The max capability to retain the previous generation velocity
        :param min_inertia: The min capability to retain the previous generation velocity
        :param implementation: Type of implementation. Default : `pswv` (Particle swarm without velocity).
                            Or else psv (Particle swarm with velocity)
        :return:
        """
        _curr_velocities = curr_velocities.copy()
        for swarm_index, swarm_config in curr_velocities.iteritems():
            for cluster_id, cluster_velocity in swarm_config.iteritems():
                local_best_pos = local_bests[swarm_index].cluster_velocity[cluster_id]
                global_best_pos = global_best.cluster_velocity[cluster_id]
                cluster_curr_pos = evaluated_swarm[swarm_index]['configuration'][cluster_id]
                if implementation == 'pswv':
                    print local_best_pos, cluster_curr_pos, global_best_pos
                    _cluster_velocity = (cognitive_factor * random() * (local_best_pos - cluster_curr_pos)) + (
                        social_factor * random() * (global_best_pos - cluster_curr_pos))
                else:
                    _cluster_velocity = (max_inertia * cluster_velocity) + (
                    cognitive_factor * random() * (local_best_pos - cluster_curr_pos)) + (
                                            social_factor * random() * (global_best_pos - cluster_curr_pos))
                _curr_velocities[swarm_index][cluster_id] = _cluster_velocity
        return _curr_velocities

    @staticmethod
    def _update_centroids(evaluated_swarm, updated_velocities):
        """ Returns updated position of each cluster for each swarm
        :param evaluated_swarm: The configuration of each swarm with each cluster inside with Base blueprint
        :param updated_velocities: The velocities updated for each cluster per each swarm
        :return: Updated coordinates for each cluster from each swarm
        """
        coordinates = defaultdict(dict)
        for swarm_index, swarm_config in evaluated_swarm.iteritems():
            for cluster_id, position in swarm_config['configuration'].iteritems():
                coordinates[swarm_index][cluster_id] = position + updated_velocities[swarm_index][cluster_id]
        return coordinates

    def _swarm_evaluation(self, association, cluster_centers, velocities):
        """
        :param association: A map of swarm indices with cluster ids and associated vectors
        :param cluster_centers: The current coordinates of the cluster
        :param velocities: The velocities for each cluster per each swarm
        :return: Return updated swarms with cluster assessment score
        """
        return {
            swarm_index: {
                'cluster_score': davies_boudin_score(self.doc_vecs, swarm_config, cluster_centers[swarm_index]),
                'configuration': cluster_centers[swarm_index], 'cluster_velocity': velocities[swarm_index],
                'association': swarm_config
            }
            for swarm_index, swarm_config in association.iteritems()}

    @staticmethod
    def update_local_conf(evaluated_swarm, local_bests):
        """
        :param evaluated_swarm: The configuration of each swarm with each cluster inside with Base blueprint
        :param local_bests: The best 
        :return:
        """
        local_bests_return = {}
        for swarm_index, state_instance in local_bests.iteritems():
            if evaluated_swarm[swarm_index]['cluster_score'] < state_instance.cluster_score:
                local_bests_return[swarm_index] = BaseState(
                    cluster_position=evaluated_swarm[swarm_index]['configuration'],
                    cluster_velocity=evaluated_swarm[swarm_index]['cluster_velocity'],
                    cluster_score=evaluated_swarm[swarm_index]['cluster_score'],
                    association=evaluated_swarm[swarm_index]['association']
                )
            else:
                local_bests_return[swarm_index] = state_instance
        return local_bests_return

    @staticmethod
    def update_global_conf(evaluated_swarm, global_best):
        """ Global best configuration achieved by a swarm
        :param evaluated_swarm: The configuration of each swarm with each cluster inside with Base blueprint
        :param global_best: The best global swarm config encountered in the swarm
        :return: Global best configuration.
        """
        for swarm_index, swarm_config in evaluated_swarm.iteritems():
            if swarm_config['cluster_score'] <= global_best.cluster_score:
                global_best = BaseState(cluster_position=swarm_config['configuration'],
                                        cluster_score=swarm_config['cluster_score'],
                                        cluster_velocity=swarm_config['cluster_velocity'],
                                        association=swarm_config['association'])
        return global_best

    def main(self, coordinates=None, velocities=None, local_bests=None, global_best=None):
        """ Main method for running pso on data vectors
        :param coordinates: The coordinates for each swarm and cluster.
        :param velocities: The velocities for each swarm and cluster.
        :param local_bests: The best configuration encountered in each index for each cluster
        :param global_best: The best global swarm config encountered in the swarm
        :return: global best configuration
        """
        for i in range(self.ite):
            print self._initialisation_boundaries(self.doc_vecs)
            exit()
            if not i:
                (coordinates, velocities) = (self.generate_initial_centroids(), self.generate_initial_velocities())
                local_bests = {
                    swarm_index: BaseState(
                        cluster_position=cluster_config,
                        cluster_velocity=velocities[swarm_index]
                    ) for swarm_index, cluster_config in coordinates.iteritems()
                    }
                global_best = choice(list(local_bests.values()))
            assignment = self.assign_to_clusters(coordinates)
            evaluated_swarm = self._swarm_evaluation(assignment, coordinates, velocities)
            local_bests = self.update_local_conf(evaluated_swarm, local_bests)
            global_best = self.update_global_conf(evaluated_swarm, global_best)
            velocities = self._update_velocities(
                                velocities, evaluated_swarm, local_bests, global_best, 1.5, 1.5, 0.72, 0.72)
            coordinates = self._update_centroids(evaluated_swarm, velocities)
        return global_best.__dict__

print PSOInstance(np.array([[1, 1], [2, 1], [10, -10], [10, 9.5]]), 100, 2, 2, 0.72, 1.5, 1.5).main()
