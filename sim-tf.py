import tensorflow as tf
import numpy as np
from collections import namedtuple
from tqdm import *

from load_materials import load_materials

tf.compat.v1.disable_eager_execution()

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
		self.blocks = tf.constant(self.blocks, dtype=tf.float32)

	def build_sim(self, iterations=10, dt=1e-8):
		# Assume for now that dt * fast_speed < self.size, otherwise our diffusion breaks.
		for i in range(3):
			assert(dt*self.FAST_SPEED < self.resolution[i])
		"""
		initial_parameters: np.array of shape (x,y,z,len(material_properties))
		"""
		self.initial_parameters = tf.compat.v1.placeholder(tf.float32, shape=self.blocks.shape[:-1])

		normed_parameters = self.initial_parameters / tf.reduce_sum(self.initial_parameters, axis=-1, keepdims=True)

		reactor = tf.reduce_sum(
			tf.reshape(normed_parameters, normed_parameters.shape+(1,)) * self.blocks,
			axis=-2
		)

		f_spont = tf.reshape(dt * self.dv * reactor[:,:,:,self.iam['spont-nprod-fission']] / 6, self.size+(1,))
		f_scatterup = tf.reshape(1-tf.exp(-dt*self.FAST_SPEED * reactor[:,:,:,self.iam['fast-scatterup']]), self.size+(1,))
		f_scatterdown = tf.reshape(1-tf.exp(-dt*self.FAST_SPEED * reactor[:,:,:,self.iam['fast-scatterdown']]), self.size+(1,))
		t_scatter = tf.reshape(1-tf.exp(-dt*self.THERMAL_SPEED * reactor[:,:,:,self.iam['thermal-scatter']]), self.size+(1,))
		f_absorb = tf.reshape(1-tf.exp(-dt*self.FAST_SPEED * reactor[:,:,:,self.iam['fast-absorb']]), self.size+(1,))
		t_absorb = tf.reshape(1-tf.exp(-dt*self.THERMAL_SPEED * reactor[:,:,:,self.iam['thermal-absorb']]), self.size+(1,))
		f_fission = tf.reshape(1-tf.exp(-dt*self.FAST_SPEED * reactor[:,:,:,self.iam['fast-nprod-fission']]), self.size+(1,))
		t_fission = tf.reshape(1-tf.exp(-dt*self.THERMAL_SPEED * reactor[:,:,:,self.iam['thermal-nprod-fission']]), self.size+(1,))
		f_out_diff_mat = tf.constant(np.repeat([dt*self.FAST_SPEED / self.resolution[i] for i in range(3)], 2).reshape((1,1,1,6)), dtype=tf.float32)
		t_out_diff_mat = tf.constant(np.repeat([dt*self.THERMAL_SPEED / self.resolution[i] for i in range(3)], 2).reshape((1,1,1,6)), dtype=tf.float32)

		# We are going to track 6 directions of neutrons in the reactor corresponding to cube faces.
		# We will assume isotropic scattering for now.
		fast_neutrons = tf.zeros(self.size+(6,), dtype=tf.float32) # (-x, +x, -y, +y, -z, +z) is the order.
		thermal_neutrons = tf.zeros(self.size+(6,), dtype=tf.float32)
		for iteration in range(iterations):
			# Generate spontaneous neutrons isotropically
			dfast = f_spont
			# Scatter fast neutrons up
			## Remove neutrons which scatter. (This is isotropic right now.)
			
			fast_scatterup = fast_neutrons * f_scatterup
			dfast += (tf.reduce_sum(fast_scatterup, axis=-1, keepdims=True) / 6) - fast_scatterup
			## Scatter fast neutrons down
			fast_scatterdown = fast_neutrons * f_scatterdown
			dthermal = tf.reduce_sum(fast_scatterdown, axis=-1, keepdims=True) / 6
			dfast -= fast_scatterdown
			# Scatter existing thermal neutrons.
			thermal_scatter = thermal_neutrons * t_scatter
			dthermal += (tf.reduce_sum(thermal_scatter, axis=-1, keepdims=True) / 6) - thermal_scatter
			
			# Absorb neutrons
			dfast -= fast_neutrons * f_absorb
			dthermal -= thermal_neutrons * t_absorb
			# Generate new neutrons via fission
			dfast += fast_neutrons * f_fission
			dfast += thermal_neutrons * t_fission
			# Finally, diffuse neutrons. Assume no external reflection of neutrons.
			## Fast
			fast_out = fast_neutrons * f_out_diff_mat
			fast_pad = tf.pad(fast_out, [(1,1)]*3+[(0,0)], mode='constant')
			diff_fast_plus = []
			for i, roll in enumerate([[-1,0,0],[1,0,0],[0,-1,0],[0,1,0],[0,0,-1],[0,0,1]]):
				diff_fast_plus.append(tf.roll(fast_pad[:,:,:,i], roll, [0,1,2]))
			dfast += tf.stack(diff_fast_plus, axis=-1)[1:-1,1:-1,1:-1]-fast_out
			## Thermal
			thermal_out = thermal_neutrons * t_out_diff_mat
			thermal_pad = tf.pad(thermal_out, [(1,1)]*3+[(0,0)], mode='constant')
			diff_thermal_plus = []
			for i, roll in enumerate([[-1,0,0],[1,0,0],[0,-1,0],[0,1,0],[0,0,-1],[0,0,1]]):
				diff_thermal_plus.append(tf.roll(thermal_pad[:,:,:,i], roll, [0,1,2]))
			dthermal += tf.stack(diff_thermal_plus, axis=-1)[1:-1,1:-1,1:-1]-thermal_out
			# New values!
			fast_neutrons += dfast
			thermal_neutrons += dthermal

		self.fast_neutrons = tf.reduce_sum(fast_neutrons, axis=-1)
		self.thermal_neutrons = tf.reduce_sum(thermal_neutrons, axis=-1)

		print('Building loss & gradient')

		# Criticality advantage function
		self.advantage = tf.math.log(tf.reduce_sum(self.fast_neutrons + self.thermal_neutrons))
		self.grad = tf.gradients(self.advantage, [self.initial_parameters])[0]

		self.sess = tf.compat.v1.Session()
		tf.compat.v1.global_variables_initializer()

		# Run once just to optimize graph a bit. (Can remove later after finished benchmarking)
		self.sess.run((self.grad, self.fast_neutrons, self.thermal_neutrons),
			feed_dict={self.initial_parameters: np.zeros(self.initial_parameters.shape)})

	def get_grad(self, initial_parameters):
		return self.sess.run(self.grad, feed_dict={self.initial_parameters: initial_parameters})

	def run_sim(self, initial_parameters):
		return self.sess.run((self.fast_neutrons, self.thermal_neutrons),
					feed_dict={self.initial_parameters: initial_parameters})


mat_properties, mat_mapping, att_mapping = load_materials('macro-simpledata.csv')
inv_mat_mapping = dict([[v,k] for k,v in mat_mapping.items()])
inv_att_mapping = dict([[v,k] for k,v in att_mapping.items()])

SIZE = tuple([12]*3)
RES = tuple([0.025]*3)
x = Reactor(mat_properties, RES, SIZE, inv_att_mapping)
x.build_sim(iterations=100, dt=5e-10)
print('Finished building simulation')


even = np.ones(SIZE+(5,))

print(x.run_sim(even)[0])
LR = 1
for i in trange(100):
	even += LR*x.get_grad(even)

#print(even)
print(x.run_sim(even)[0])

"""
tamperless, tamperfull = np.zeros(SIZE+(5,)), np.zeros(SIZE+(5,))
tamperfull[:,:,:,2]=1
tamperfull[1:-1,1:-1,1:-1,2]=0
tamperfull[1:-1,1:-1,1:-1,-1]=1
tamperless[1:-1,1:-1,1:-1,-1]=1

fast, thermal = x.run_sim(tamperless)
#print(fast)
#print('-----')
#print(thermal)
print('-----')
print(fast[1,1,1], fast[SIZE[0]//2, SIZE[1]//2, SIZE[2]//2])
print(thermal[1,1,1], thermal[SIZE[0]//2, SIZE[1]//2, SIZE[2]//2])


fast, thermal = x.run_sim(tamperfull)
#print(fast)
#print('-----')
#print(thermal)
print('-----')
print(fast[1,1,1], fast[SIZE[0]//2, SIZE[1]//2, SIZE[2]//2])
print(thermal[1,1,1], thermal[SIZE[0]//2, SIZE[1]//2, SIZE[2]//2])
"""