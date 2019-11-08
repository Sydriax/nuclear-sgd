import tensorflow as tf
import numpy as np
from collections import namedtuple

from load_materials import load_materials

"""
Reactor is defined by a grid of materials.

Some spaces may be pre-defined, others modifiable.

Init:
- Materials property dicts
- Initial materials.
- Which is modifiable
- Loss: fn of form (materials_grid, final_neutron, <final_temp>, non-homog penalty)
	(returns) tf loss.

Optimize:
- Epochs
- Non-homogeneity penalty schedule.

"""



# Build basic blocks



class Reactor():
	def __init__(self, mat_properties, resolution, size, inv_att_mapping):
		"""
		material_properties: dict of appropriate properties
		Resolution: (dx,dy,dz)
		Size: (x, y, z)
		initial_parameters: np.array of shape (x,y,z,len(material_properties)-1)
		"""
		self.mat_properties = mat_properties
		self.resolution = resolution
		self.iam = inv_att_mapping
		self.dv = resolution[0]*resolution[1]*resolution[2]
		self.size = size

		self.THERMAL_SPEED = 2.2e3
		self.FAST_SPEED = 14e6 # Not sure what to do about this...
		# For now will just assume all surviving fast neutrons move every timestep.

		self.blocks = self.mat_properties.reshape((1,1,1)+self.mat_properties.shape)
		self.blocks = np.repeat(self.blocks, self.size[0], axis=0)
		self.blocks = np.repeat(self.blocks, self.size[1], axis=1)
		self.blocks = np.repeat(self.blocks, self.size[2], axis=2)

	def run_simulation(self, initial_parameters, iterations=10, dt=0.0001):
		"""
		initial_parameters: np.array of shape (x,y,z,len(material_properties))
		"""
		reactor = np.sum(
			initial_parameters.reshape(initial_parameters.shape+(1,)) * self.blocks,
			axis=-2
		)

		# We are going to track 6 directions of neutrons in the reactor corresponding to cube faces.
		# We will assume isotropic scattering for now.
		fast_neutrons = np.zeros(self.size+(6,))
		thermal_neutrons = np.zeros(self.size+(6,))
		for iteration in range(iterations):
			# Generate spontaneous neutrons isotropically
			spont_neutrons = dt * self.dv * reactor[:,:,:,self.iam['spont-nprod-fission']] / 6



mat_properties, mat_mapping, att_mapping = load_materials('simpledata.csv')
inv_mat_mapping = dict([[v,k] for k,v in mat_mapping.items()])
inv_att_mapping = dict([[v,k] for k,v in att_mapping.items()])
x = Reactor(mat_properties, None, (2,3,4), inv_att_mapping)
x.run_simulation(np.ones((2,3,4,4))/4)
#print(x.blocks.shape)