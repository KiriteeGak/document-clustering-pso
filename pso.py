import numpy as np

class pso_instance(object):
	def __init__(self, w2v_model, n_particles, n_clusters, inertia, local_accel_coeff, global_accel_coeff):
		self.doc_vecs = w2v_model
		self.n_particles = n_particles
		self.n_clusters = n_clusters
		self.inertia = inertia
		self.lac = local_accel_coeff
		self.gac = global_accel_coeff
