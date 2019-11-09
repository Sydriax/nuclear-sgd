import tensorflow as tf
import numpy as np
from collections import namedtuple
from tqdm import *

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
		self.FAST_SPEED = 1.4e7 # Not sure what to do about this...
		# For now will just assume all surviving fast neutrons move every timestep.

		self.blocks = self.mat_properties.reshape((1,1,1)+self.mat_properties.shape)
		self.blocks = np.repeat(self.blocks, self.size[0], axis=0)
		self.blocks = np.repeat(self.blocks, self.size[1], axis=1)
		self.blocks = np.repeat(self.blocks, self.size[2], axis=2)

	def run_simulation(self, initial_parameters, iterations=10, dt=1e-8):
		# Assume for now that dt * fast_speed < self.size, otherwise our diffusion breaks.
		for i in range(3):
			assert(dt*self.FAST_SPEED < self.resolution[i])
		"""
		initial_parameters: np.array of shape (x,y,z,len(material_properties))
		"""
		reactor = np.sum(
			initial_parameters.reshape(initial_parameters.shape+(1,)) * self.blocks,
			axis=-2
		)

		# We are going to track 6 directions of neutrons in the reactor corresponding to cube faces.
		# We will assume isotropic scattering for now.
		fast_neutrons = np.zeros(self.size+(6,)) # (-x, +x, -y, +y, -z, +z) is the order.
		thermal_neutrons = np.zeros(self.size+(6,))
		for iteration in trange(iterations):
			dfast = np.zeros(self.size+(6,))
			dthermal = np.zeros(self.size+(6,))
			# Generate spontaneous neutrons isotropically
			dfast += (dt * self.dv * reactor[:,:,:,self.iam['spont-nprod-fission']] / 6).reshape(self.size+(1,))
			# Scatter fast neutrons up
			## Remove neutrons which scatter. (This is isotropic right now.)
			
			fast_scatterup = fast_neutrons * (1-np.exp((-dt*self.FAST_SPEED * reactor[:,:,:,self.iam['fast-scatterup']]))).reshape(self.size+(1,))
			dfast += (np.sum(fast_scatterup, axis=-1, keepdims=True) / 6) - fast_scatterup
			## Scatter fast neutrons down
			fast_scatterdown = fast_neutrons * (1-np.exp((-dt*self.FAST_SPEED * reactor[:,:,:,self.iam['fast-scatterdown']]))).reshape(self.size+(1,))
			dthermal += np.sum(fast_scatterdown, axis=-1, keepdims=True) / 6
			dfast -= fast_scatterdown
			# Scatter existing thermal neutrons.
			thermal_scatter = thermal_neutrons * (1-np.exp((-dt*self.THERMAL_SPEED * reactor[:,:,:,self.iam['thermal-scatter']]))).reshape(self.size+(1,))
			dthermal += (np.sum(thermal_scatter, axis=-1, keepdims=True) / 6) - thermal_scatter
			
			# Absorb neutrons
			dfast -= fast_neutrons * (1-np.exp((-dt*self.FAST_SPEED * reactor[:,:,:,self.iam['fast-absorb']])).reshape(self.size+(1,)))
			dthermal -= thermal_neutrons * (1-np.exp((-dt*self.THERMAL_SPEED * reactor[:,:,:,self.iam['thermal-absorb']])).reshape(self.size+(1,)))
			# Generate new neutrons via fission
			dfast += fast_neutrons * (1-np.exp((-dt*self.FAST_SPEED * reactor[:,:,:,self.iam['fast-nprod-fission']])).reshape(self.size+(1,)))
			dfast += thermal_neutrons * (1-np.exp((-dt*self.THERMAL_SPEED * reactor[:,:,:,self.iam['thermal-nprod-fission']])).reshape(self.size+(1,)))
			# Finally, diffuse neutrons. Assume no external reflection of neutrons.
			## Fast
			fast_out = fast_neutrons * np.repeat([dt*self.FAST_SPEED / self.resolution[i] for i in range(3)], 2).reshape((1,1,1,6))
			fast_pad = np.pad(fast_out, [(1,1)]*3+[(0,0)], mode='constant')
			diff_fast_plus = []
			for i, roll in enumerate([[-1,0,0],[1,0,0],[0,-1,0],[0,1,0],[0,0,-1],[0,0,1]]):
				diff_fast_plus.append(np.roll(fast_pad[:,:,:,i], roll, [0,1,2]))
			dfast += np.stack(diff_fast_plus, axis=-1)[1:-1,1:-1,1:-1]-fast_out
			## Thermal
			thermal_out = thermal_neutrons * np.repeat([dt*self.THERMAL_SPEED / self.resolution[i] for i in range(3)], 2).reshape((1,1,1,6))
			thermal_pad = np.pad(thermal_out, [(1,1)]*3+[(0,0)], mode='constant')
			diff_thermal_plus = []
			for i, roll in enumerate([[-1,0,0],[1,0,0],[0,-1,0],[0,1,0],[0,0,-1],[0,0,1]]):
				diff_thermal_plus.append(np.roll(thermal_pad[:,:,:,i], roll, [0,1,2]))
			dthermal += np.stack(diff_thermal_plus, axis=-1)[1:-1,1:-1,1:-1]-thermal_out
			# New values!
			fast_neutrons += dfast
			thermal_neutrons += dthermal

		return np.sum(fast_neutrons,axis=-1), np.sum(thermal_neutrons, axis=-1)


mat_properties, mat_mapping, att_mapping = load_materials('macro-simpledata.csv')
inv_mat_mapping = dict([[v,k] for k,v in mat_mapping.items()])
inv_att_mapping = dict([[v,k] for k,v in att_mapping.items()])

SIZE = tuple([12]*3)
RES = tuple([0.025]*3)
x = Reactor(mat_properties, RES, SIZE, inv_att_mapping)
tamperless, tamperfull = np.zeros(SIZE+(5,)), np.zeros(SIZE+(5,))
tamperfull[:,:,:,2]=1
tamperfull[1:-1,1:-1,1:-1,2]=0
tamperfull[1:-1,1:-1,1:-1,-1]=1
tamperless[1:-1,1:-1,1:-1,-1]=1

fast, thermal = x.run_simulation(tamperless, iterations=2000, dt=5e-10)
#print(fast)
#print('-----')
#print(thermal)
print('-----')
print(fast[1,1,1], fast[SIZE[0]//2, SIZE[1]//2, SIZE[2]//2])
print(thermal[1,1,1], thermal[SIZE[0]//2, SIZE[1]//2, SIZE[2]//2])


fast, thermal = x.run_simulation(tamperfull, iterations=2000, dt=5e-10)
#print(fast)
#print('-----')
#print(thermal)
print('-----')
print(fast[1,1,1], fast[SIZE[0]//2, SIZE[1]//2, SIZE[2]//2])
print(thermal[1,1,1], thermal[SIZE[0]//2, SIZE[1]//2, SIZE[2]//2])