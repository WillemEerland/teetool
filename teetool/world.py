# "main" source file for trajectory analysis in this package. Clusters of trajectories are added here, modelled accordingly (whatever settings desired), and visualised in whatever is desired (single / multiple clusters)

import helpers
import model

# import support files here
import numpy as np
import mayavi.mlab as mlab

class World(object):
	"""
	This class provides the interface for the trajectory analysis tool.

	<description>

	Initialisation arguments:
	 - name
	 - dimension

	Properties:
	 -
	 -
	"""

	def __init__(self, name="", dimension=3):
		"""
		<description>
		"""

		# validate name
		if type(name) is not str:
			raise TypeError("expected string, not {}".format(type(name)))

		# validate dimension
		if type(dimension) is not int:
			raise TypeError("expected integer, not {}".format(type(dimension)))

		if (dimension != 2) and (dimension != 3):
			raise ValueError("expected dimensionality 2 or 3, not {}".format(dimension))

		# set values
		self.name = name
		self.D = dimension

		# initial parameters
		self.clusters = [] # list holding clusters

	def overview(self):
		"""
		prints overview in console
		"""

		print("*** overview [{}] ***".format(self.name))

		for (i, this_cluster) in enumerate(self.clusters):
			has_model = "*"
			if (this_cluster["model"] == None):
				has_model = "-"
			print("{0} [{1}] [{2}]".format(i, this_cluster["name"], has_model))

		return True

	def addCluster(self, cluster_data, cluster_name=""):
		"""
		<description>

		Input arguments:
			- aCluster: list with tuples (x, Y) representing trajectory data
			- name: a string with the name of the cluster
		"""

		# validate cluster_name
		if type(cluster_name) is not str:
			raise TypeError("expected string, not {}".format(type(cluster_name)))

		# validate cluster_data
		if type(cluster_data) is not list:
			raise TypeError("expected list, not {}".format(type(cluster_data)))

		# validate trajectory_data
		for (i, trajectory_data) in enumerate(cluster_data):
			# check type
			if type(trajectory_data) is not tuple:
				raise TypeError("expected tuple, item {0} is a {1}".format(i,type(trajectory_data)))
			# check values x [M x 1], Y [M x D]
			(x, Y) = trajectory_data
			(M, D) = Y.shape
			if (D != self.D):
				raise ValueError("dimension not correct")
			if (M != np.size(x,0)):
				raise ValueError("number of data-points do not match")

		# add new cluster [ holds "name" and "data" ]
		new_cluster = {}

		new_cluster["name"] = cluster_name
		new_cluster["data"] = cluster_data
		new_cluster["model"] = None # initialise empty model

		self.clusters.append(new_cluster)

		return True

	def model(self):
		"""
		<description>
		"""

		# TODO allow custom input
		settings = {}
		settings["model_type"] = "resample"
		settings["mpoints"] = 100

		for (i, this_cluster) in enumerate(self.clusters):

			# build a new model
			new_model = model.Model(this_cluster["data"], settings)

			# overwrite value
			this_cluster["model"] = new_model
			self.clusters[i] = this_cluster

	def show_trajectories(self):
		"""
		visualise trajectories using mayavi
		"""

		nclusters = len(self.clusters)  # number of clusters
		colours = helpers.getDistinctColours(nclusters)  # colours

		mlab.figure()

		for (i, this_cluster) in enumerate(self.clusters):
			# this cluster
			cluster_data = this_cluster["data"]
			for (x, Y) in cluster_data:
				# this trajectory
				mlab.plot3d(Y[:, 0], Y[:, 1], Y[:, 2], color=colours[i], tube_radius=.1)

		mlab.show()

		return True

	def show_model(self):
		"""
		<description>
		"""

		# TODO allow inputs

		x, y, z = np.ogrid[-60:60:20j, -10:240:20j, -60:60:20j]

		s_models = []

		for (i, this_cluster) in enumerate(self.clusters):
			# evaluate model on grid
			s = this_cluster["model"].eval(x, y, z)
			# store
			s_models.append(s)

		# TODO HARDCODE TEST
		s = s_models[0] + s_models[1]

		# normalise
		s = (s - np.min(s)) / (np.max(s) - np.min(s))

		# mayavi
		src = mlab.pipeline.scalar_field(s)
		mlab.pipeline.iso_surface(src, contours=[s.min()+0.3*s.ptp(), ], opacity=0.2)
		mlab.pipeline.volume(mlab.pipeline.scalar_field(s), vmin=.2, vmax=.8)
		mlab.outline()
		mlab.show()
