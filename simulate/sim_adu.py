import numpy as np
import os
from spipy.analyse import rotate
from scipy.linalg import get_blas_funcs
import numexpr as ne

def help(module):
	if module=="multi_process":
		print("This function generates spi patterns by simulating atom reflection using multiple jobs")
		print("    -> Input: save_dir (path to save simulation result, a directory path)")
		print("              pdb_file (path of your input pdb file)")
		print("              param (simulation parameters, the same with 'spipy.simulate.sim' module)")
		print("     #option: euler_mode (how to genreate euler angles, str, chosen from 'random', 'helix' and 'predefined', default='random')")
		print("     #option: euler_order (rotation order, such as 'zxz' or 'zyz'..., default='zxz')")
		print("     #option: euler_range (the range of euler angles to rotate object, numpy.array([[alpha_min,alpha_max],[beta_min,beta_max],[gamma_min,gamma_max]]), shape=(3,2)), default=None")
		print("     #option: predefined (if euler_mode is 'predefined', then you have to specify all euler angles used for object rotation, shape=(Ne,3), default=None)")
		print("     #option: njobs (number of jobs, int, default=2)")
		print("    -> NO RETURN")
	elif module=="single_process":
		print("This function generates spi patterns by simulating atom reflection using single jobs")
		print("    -> Input: pdb_file (path of your input pdb file)")
		print("              pdb_file (path of your input pdb file)")
		print("              param (simulation parameters, the same with 'spipy.simulate.sim' module)")
		print("     #option: euler_mode (how to genreate euler angles, str, chosen from 'random', 'helix' and 'predefined', default='random')")
		print("     #option: euler_order (rotation order, such as 'zxz' or 'zyz'..., default='zxz')")
		print("     #option: euler_range (the range of euler angles to rotate object, numpy.array([[alpha_min,alpha_max],[beta_min,beta_max],[gamma_min,gamma_max]]), shape=(3,2)), default=None")
		print("     #option: predefined (if euler_mode is 'predefined', then you have to specify all euler angles used for object rotation, shape=(Ne,3), default=None)")
		print("     #option: save_dir (path to save simulation result, a directory path, default=None and the result will be returned)")
		print("    -> Return: dataset (a dict containing all information and result of the simulation, only if you set 'save_dir=None')")
	else:
		raise ValueError("No module names "+str(module))


class simulation():

	config_param = {'parameters|detd' : 200, 'parameters|lambda' : 2.5, \
					'parameters|detsize' : 128, 'parameters|pixsize' : 0.3, \
					'parameters|stoprad' : 0, 'parameters|polarization' : 'x', \
					'make_data|num_data' : 100, 'make_data|fluence' : 1e14, \
					'make_data|scatter_factor' : False, 'make_data|ram_first' : True}
	euler = None  # alpha 0,2pi ; beta 0,pi ; gamma 0,2pi  shape=(Ne,3) [intrisinc]
	order = None  # rotation order 'zxz','zyz',...
	atoms = {'coordinate':None, 'index':None} # (angstrom) coordinate.shape=(Na,3)
	sample_size = None  # (nanometer)
	oversampl = None
	# self.k0 , self.screen_x

	def configure(self, pdb_file, param):
		from spipy.analyse import q
		from spipy.image import io
		# read parameters
		for key, value in param.iteritems():
			if key in self.config_param.keys():
				if type(value)==type(self.config_param[key]):
					self.config_param[key] = value
				else:
					raise ValueError("The data type of '"+key+"' in param is wrong")
			else:
				raise ValueError('param has unknown items')
		self.k0 = np.array([0,0,1])
		if self.config_param['parameters|polarization']=='x':
			self.screen_x = np.array([1,0,0])
		else:
			self.screen_x = np.array([0,1,0])
		# read pdb
		if os.path.exists(pdb_file):
			atom_info = io._readpdb(pdb_file)
			self.atoms['coordinate'] = atom_info[:,1:4]
			self.atoms['index'] = atom_info[:,0]
		else:
			raise ValueError('pdb file path is invalid')
		# put molecular to the coordinate origin
		center = self.atoms['coordinate'].mean(axis=0)
		self.atoms['coordinate'] -= center
		# calculate oversampling rate
		self.sample_size = np.linalg.norm(self.atoms['coordinate'], axis=1).max()*2/10.  # in nanometer
		self.oversampl = q.oversamp_rate(self.sample_size, self.config_param['parameters|detd'], \
				self.config_param['parameters|lambda'], self.config_param['parameters|pixsize'])

	def generate_euler(self, mode='random', order='zxz', arange=None, predefined=None):
		if arange is None:
			if mode=='random':
				pat_num = self.config_param['make_data|num_data']
				alpha = np.random.rand(pat_num) * 2 * np.pi
				beta = np.random.rand(pat_num) * np.pi
				gamma = np.random.rand(pat_num) * 2 * np.pi
				self.euler = np.vstack([alpha,beta,gamma]).T
			elif mode=='helix':
				pat_num = self.config_param['make_data|num_data']
				alpha = np.linspace(0,np.pi*2,pat_num)
				beta = np.linspace(0,np.pi,pat_num)
				gamma = np.linspace(0,np.pi*2,pat_num)
				self.euler = np.vstack([alpha,beta,gamma]).T
			elif mode=='predefined':
				if predefined is None:
					raise RuntimeError("I need euler angles you defined in mode 'predefined'")
				elif len(predefined.shape)!=2 or predefined.shape[1]!=3:
					raise RuntimeError("I can't recognize your input euler angles ...")
				else:
					self.euler = predefined
					self.config_param['make_data|num_data'] = predefined.shape[0]
			else:
				raise RuntimeError("I can't recognize your mode")
		elif arange.shape==(3,2):
			if mode=='random':
				pat_num = self.config_param['make_data|num_data']
				alpha = np.random.rand(pat_num) * (arange[0].max()-arange[0].min()) + arange[0].min()
				beta = np.random.rand(pat_num) * (arange[1].max()-arange[1].min()) + arange[1].min()
				gamma = np.random.rand(pat_num) * (arange[2].max()-arange[2].min()) + arange[2].min()
				self.euler = np.vstack([alpha,beta,gamma]).T
			elif mode=='helix':
				pat_num = self.config_param['make_data|num_data']
				alpha = np.linspace(arange[0].min(),arange[0].max(),pat_num+2)[1:-1]
				beta = np.linspace(arange[1].min(),arange[1].max(),pat_num+2)[1:-1]
				gamma = np.linspace(arange[2].min(),arange[2].max(),pat_num+2)[1:-1]
				self.euler = np.vstack([alpha,beta,gamma]).T
			else:
				raise RuntimeError("I can't recognize your mode")
		else:
			raise RuntimeError('Your input euler angle range is invalid')
		if len(self.euler) == self.config_param['make_data|num_data']:
			self.order = order
		else:
			raise RuntimeError("Here occurs some error in configuration ... error-code:s2_geneul. Please report this bug")

	def generate_box(self, den_map):
		if den_map is None:
			pass
		elif len(self.den_map.shape)==3:
			center = (np.array(den_map.shape)+1)/2
			box_size = (np.array(den_map.shape)*self.oversampl).astype(int)
			box = np.zeros(box_size)
			box_cen = (np.array(box.shape)+1)/2
			box[box_cen[0]-center[0]:box_cen[0]+density_map.shape[0]-center[0],\
				box_cen[1]-center[1]:box_cen[1]+density_map.shape[1]-center[1],\
				box_cen[2]-center[2]:box_cen[2]+density_map.shape[2]-center[2]] = den_map
			return box
		else:
			raise RuntimeError('Your input density map or oversampleing rate is invalid')

	def get_scatt(self, atom_index, pix_r):
		'''
		return a lookup table, shape=(max(atom_index)+1,max(pix_r)+1)
		[NOTICE!] for those atom index or q that are not used in simulation, the value of corresponding cell
		of the return table is ZERO ! AND, the atom index start from 1 (which means the first row of the
		returned table is always zero)!
		'''
		def gaussian(a, b, c, q):
			a = a.reshape((len(a),1))
			b = b.reshape((len(b),1))
			c = c.reshape((len(c),1))
			q = q.reshape((len(q),))
			return a*np.exp(-b*q**2)+c

		det_lambda = self.config_param['parameters|lambda']
		det_d = self.config_param['parameters|detd']
		det_ps = self.config_param['parameters|pixsize']

		abs_index = np.sort(list(set(atom_index))).astype(int)
		abs_r = np.sort(list(set(pix_r))).astype(int)
		q = np.sin(np.arctan(abs_r*det_ps/det_d)/2)/det_lambda
		scatter_file = os.path.join(os.path.dirname(__file__), 'aux/scattering.npy')
		scatterf = np.load(scatter_file)[()]
		abc = np.zeros((len(abs_index), 9))
		for ind,k in enumerate(abs_index):
			abc[ind] = scatterf[k]
		gau_1 = gaussian(abc[:,0],abc[:,1],abc[:,8],q)
		gau_2 = gaussian(abc[:,2],abc[:,3],abc[:,8],q)
		gau_3 = gaussian(abc[:,4],abc[:,5],abc[:,8],q)
		gau_4 = gaussian(abc[:,6],abc[:,7],abc[:,8],q)
		fq = gau_1 + gau_2 + gau_3 + gau_4
		Fq = np.zeros((np.max(abs_index)+1,np.max(abs_r)+1))
		Fq[abs_index.reshape((len(abs_index),1)),abs_r] = fq
		return Fq

	def rotate_mol(self, euler_angle):
		if self.order is None or self.atoms['index'] is None:
			raise RuntimeError('Please configure and generate euler first!')
		new_atoms = {}
		new_atoms['index'] = self.atoms['index']
		old_coor = self.atoms['coordinate']
		rotate_matrix = rotate.eul2rotm(euler_angle, self.order)
		new_coor = rotate_matrix.dot(old_coor.T).T
		new_atoms['coordinate'] = new_coor
		return new_atoms

	def generate_pattern(self, euler_angles, patt):
		import psutil
		import time
		import gc

		if self.order is None or self.atoms['index'] is None:
			raise RuntimeError('Please configure and generate euler first!')
		# get parameters
		det_l = self.config_param['parameters|detsize']
		det_d = np.float32(self.config_param['parameters|detd'])
		det_ps = np.float32(self.config_param['parameters|pixsize'])
		det_lambda = np.float32(self.config_param['parameters|lambda'])
		ati = self.atoms['index']
		ati = ati.reshape((len(ati),1)).astype(int)
		# calculate detector information
		det_cen = (det_l-1)/2.0
		detx, dety = np.mgrid[0:det_l,0:det_l]        # here in pixel
		detx = detx.flatten()
		dety = dety.flatten()
		if self.config_param['make_data|scatter_factor'] is True:
			pix_r = np.sqrt((detx-det_cen)**2 + (dety-det_cen)**2).astype(int)
			scatt = self.get_scatt(self.atoms['index'], pix_r)  # scattering factor, shape=(max(atom_index)+1,max(pix_r)+1)
		else:
			scatt = ati
		screen_xy = np.array([detx-det_cen, dety-det_cen]).T * det_ps   # here in mm
		screen_xyz = np.hstack([screen_xy, np.zeros((det_l**2,1))+det_d])        # here in mm
		k_prime = screen_xyz/np.linalg.norm(screen_xyz,axis=1).reshape((len(screen_xyz),1)) # here no unit
		dk = (k_prime - self.k0)/det_lambda   # dk.shape=(Nk,3) the value of dk of every pixel on detector, angstrom^(-1)

		# now start loop
		for ind,euler_angle in enumerate(euler_angles):
			print('\ngenerating '+str(ind+1)+'th patterns')
			time0 = time.time()
			Natoms = self.rotate_mol(euler_angle)
			dr = Natoms['coordinate']    # dr.shape=(Nr,3) the position vector of atoms, in angstrom

			pat = np.zeros(len(dk),dtype=np.complex64)
			if self.config_param['make_data|ram_first']:
				# forced to use cblas lib to accelerate
				gemm = get_blas_funcs("gemm",[dr,dk[0].reshape((3,1))])
				for ii,dkk in enumerate(dk):
					temp = np.complex128(1j)*-1*gemm(1,dr,dkk.reshape((3,1)))    # matrix shape=(Nr,1)
					temp = ne.evaluate('exp(temp)')
					if self.config_param['make_data|scatter_factor'] is True:
						temp *= scatt[ati,pix_r[ii]].reshape(temp.shape)
					else:
						temp *= scatt
					pat[ii] = np.sum(temp)
				print u"RAM (MB): ",psutil.Process(os.getpid()).memory_info().rss/1024.0**2
			else:
				# forced to use cblas lib to accelerate
				gemm = get_blas_funcs("gemm",[dr.astype(np.float32),dk.T.astype(np.float32)])
				pat = np.complex128(1j)*-1*gemm(1,dr,dk.T)    # matrix shape=(Nr,Nk)
				pat = ne.evaluate('exp(pat)')

				if self.config_param['make_data|scatter_factor'] is True:
					pat *= scatt[ati, pix_r]
				else:
					pat *= scatt
				print u"RAM (MB): ",psutil.Process(os.getpid()).memory_info().rss/1024.0**2
				pat = np.sum(pat,axis=0)
			pat = pat.reshape((det_l, det_l))
			pat = np.abs(pat)**2
			# evaluate adu
			adu = np.float32(self.config_param['make_data|fluence'] * 12398.419 / det_lambda / 1e12)
			patt[ind] = pat * adu/np.sum(pat)
			del pat
			gc.collect()
			print('Done. Time (s): '+str(time.time()-time0))
		return patt

	def assemble(self):
		if self.order is None or self.oversampl is None:
			raise RuntimeError('Please configure and generate euler first!')
		from spipy.image import radp

		num_pat = self.config_param['make_data|num_data']
		det_l = self.config_param['parameters|detsize']
		bstopR = self.config_param['parameters|stoprad']
		patterns = np.zeros((num_pat, det_l, det_l))
		# generate patterns
		self.generate_pattern(self.euler, patterns)
		beam_stop = radp.circle(2, bstopR) + (np.array(patterns.shape[1:])-1)/2
		if beam_stop is not None:
			patterns[:,beam_stop[:,0],beam_stop[:,1]] = 0
		output = {'oversampling_rate' : self.oversampl, \
					'euler_angles' : self.euler, \
					'rotation_order' : self.order, \
					'simu_parameters' : self.config_param, \
					'patterns' : patterns}
		return output

def single_process(pdb_file, param, euler_mode='random', euler_order='zxz', euler_range=None, predefined=None, save_dir=None):
	import h5py
	if save_dir is not None and not os.path.isdir(save_dir):
		raise ValueError('Your save directory is invalid')
	sim = simulation()
	sim.configure(pdb_file, param)
	sim.generate_euler(euler_mode, euler_order, euler_range, predefined)
	dataset = sim.assemble()
	if save_dir is not None:
		# save h5
		savef = h5py.File(os.path.join(save_dir, 'spipy_adu_simulation.h5'), 'w')
		savef.create_dataset('oversampling_rate', data=dataset['oversampling_rate'])
		savef.create_dataset('rotation_order', data=dataset['rotation_order'])
		savef.create_dataset('patterns', data=dataset['patterns'], chunks=True, compression="gzip")
		savef.create_dataset('euler_angles', data=dataset['euler_angles'], chunks=True, compression="gzip")
		grp = savef.create_group('simu_parameters')
		for k, v in dataset['simu_parameters'].iteritems():
			grp.create_dataset(k, data=v)
		savef.close()
		return
	else:
		return dataset

def multi_process(save_dir, pdb_file, param, euler_mode='random', euler_order='zxz', euler_range=None, predefined=None, njobs=2):
	import h5py
	import multiprocessing as mp
	if not os.path.isdir(save_dir):
		raise ValueError('Your save directory is invalid')
	if njobs<1:
		raise ValueError('Number of jobs should not <1')
	if predefined is not None:
		euler_mode = 'predefined'
	# multi-process
	pool = mp.Pool(processes = njobs)
	result = []
	if euler_mode=='predefined':
		num_pat = len(predefined)
	else:
		num_pat = param['make_data|num_data']
	pool_bin = np.linspace(0, num_pat, njobs+1, dtype=int)
	patterns = np.zeros((num_pat, param['parameters|detsize'], param['parameters|detsize']))
	eul_angles = np.zeros((num_pat, 3))
	# split euler
	if euler_mode=='random' or euler_mode=='helix':
		if euler_range is not None:
			alpha_bin = np.linspace(euler_range[0,0], euler_range[0,1], njobs+1)
			beta_bin = np.linspace(euler_range[1,0], euler_range[1,1], njobs+1)
			gamma_bin = np.linspace(euler_range[2,0], euler_range[2,1], njobs+1)
		else:
			alpha_bin = np.linspace(0, np.pi*2, njobs+1)
			beta_bin = np.linspace(0, np.pi, njobs+1)
			gamma_bin = np.linspace(0, np.pi*2, njobs+1)
	for ind, i in enumerate(pool_bin[:-1]):
		print('Start process ' + str(ind))
		param_copy = param.copy()
		param_copy['make_data|num_data'] = int(pool_bin[ind+1]-i)
		if euler_mode=='random' or euler_mode=='helix':
			euler_range_copy = np.array([alpha_bin[ind:ind+2], beta_bin[ind:ind+2], gamma_bin[ind:ind+2]])
			euler_defined_copy = None
		elif euler_mode=='predefined':
			euler_defined_copy = predefined[i:pool_bin[ind+1]]
			euler_range_copy = None
		result.append(pool.apply_async(single_process, args=(pdb_file, param_copy, euler_mode, euler_order, euler_range_copy, euler_defined_copy, None,)))
	pool.close()
	pool.join()
	# merge
	oversampling_rate = None
	for ind,re in enumerate(result):
		datset = re.get()
		patterns[pool_bin[ind]:pool_bin[ind+1]] = datset['patterns']
		eul_angles[pool_bin[ind]:pool_bin[ind+1]] = datset['euler_angles']
		oversampling_rate = datset['oversampling_rate']
	# save h5
	savef = h5py.File(os.path.join(save_dir, 'spipy_adu_simulation.h5'), 'w')
	savef.create_dataset('oversampling_rate', data=oversampling_rate)
	savef.create_dataset('rotation_order', data=euler_order)
	savef.create_dataset('patterns', data=patterns, chunks=True, compression="gzip")
	savef.create_dataset('euler_angles', data=eul_angles, chunks=True, compression="gzip")
	grp = savef.create_group('simu_parameters')
	for k, v in param.iteritems():
		grp.create_dataset(k, data=v)
	savef.close()
