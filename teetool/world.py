# "main" source file for trajectory analysis in this package. Clusters of trajectories are added here, modelled accordingly (whatever settings desired), and visualised in whatever is desired (single / multiple clusters)


# import support files here
import numpy as np

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