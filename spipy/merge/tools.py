import numpy as np
from ..image import quat
from ..analyse import rotate
import numexpr as ne
from scipy.linalg import get_blas_funcs
import sys


def help(module):
	if module == "get_slice":
		print("Generate slices from a 3D model according to given rotations")
		print("    -> Input : model ( the model, a 3D numpy array )")
		print("               rotations ( a list of quaternions or euler angles, [[w,qx,qy,qz],[alpha,beta,gamma]...] ) ")
		print("               det_size ( the size of generated patterns (in pixels), [Npx, Npy] )")
		print("     *option : det_center ( the center of generated patterns (in pixels), default=None and the geometry center is used )")
		print("     *option : mask ( pattern mask , 2d numpy array where 1 means masked area and 0 means useful area, default is None )")
		print("     *option : slice_coor_ori ( well-mapped 3D coordinates for pixels in detector without rotation, shape=(3,N), N=det_size[0]*det_size[1], default is None )")
		print("     *option : euler_order ( rotation order if you are using euler angles, default is 'zxz' )")
		print("    -> Output: slices, shape=(Nq, Npx, Npy) , Nq is the number of rotations")
	elif module == "merge_slice":
		print("Merge sereval slices into a 3D model according to given rotations")
		print("    -> Input : model ( the model, a 3D numpy array )")
		print("               rotations ( a list of quaternions or euler angles, [[w,qx,qy,qz],[alpha,beta,gamma]...] ) ")
		print("               slices ( slices to merge into model, numpy array, shape=(N,Npx,Npy) )")
		print("     *option : weights ( inital interpolation weights for every pixel of input model, shape=model.shape, default is None and weights=ones is used )")
		print("     *option : det_center ( the center of generated patterns (in pixels), default=None and the geometry center is used )")
		print("     *option : mask ( pattern mask , 2d numpy array where 1 means masked area and 0 means useful area, default is None )")
		print("     *option : slice_coor_ori ( well mapped 3D coordinates for pixels in detector without rotation, shape=(3,N), N=det_size[0]*det_size[1], default is None )")
		print("     *option : euler_order ( rotation order if you are using euler angles, default is 'zxz' )")
		print("    -> Output: None, model is modified directly")
	elif module == "poisson_likelihood":
		print("Calculate poisson likelihood between a model slice and exp pattern")
		print("    -> Input : W_j ( model slice in orientation j, numpy 1d/2d array, do masking in advance )")
		print("               K_k ( experiment pattern, numpy 1d/2d array, do masking in advance )")
		print("     *option : beta ( float, suggested values are from 1 to 50, default=1 )")
		print("     *option : weight ( float, the weight of orientation j, if orientations are not strictly uniformly sampled, default is None )")
		print("    -> Output: float, R_jk = weight * Product{W_j**K_k*exp(-W_j)} ** beta")
	elif module == "maximization":
		print("Calculate updated slice of orientation j after likelihood maximization")
		print("    -> Input : K_ks ( all useful experiment patterns, numpy array, shape=(N,Np), reshape pattern to array or do masking in advance !)")
		print("               Prob_ks ( probabilities of all useful patterns (after normalizing in every orientation) in orientation j, shape=(N,) )")
		print("    -> Output: W_j_prime, updated slice in orientation j (flattened), length = K_ks.shape[1]")
	elif module == 'get_quaternion':
		print("Calculate quaternions which are uniformly distributed in orientation space (sampling weights is uniform)")
		print("    -> Input : Num_level ( int, number of output quaternions is 2*Num_level^3 )")
		print("    -> Output: quaternions, numpy.array, shape=(2*Num_level^3,4)")
	else:
		raise ValueError("No module names "+str(module))



def get_slice(model, rotations, det_size, det_center=None, \
					mask=None, slice_coor_ori=None, euler_order="zxz"):
	'''
	get one slice from a 3D matrix (model), whose orientation depends on a quaternion
	model        : 3d numpy array
	rotations    : 2d list, [[w, qx, qy, qz], [alpha, beta, gamma], ...], length = Nq
	               [NOTICE] for euler angle, the default order is intrinsic 'zxz'
	det_size     : [size_x, size_y]
	det_center   : [cx, cy] or None, start from 0
	mask         : np.array, shape=(Nx,Ny), 1 means masked pixel
	slice_coor_ori : well-mapped 3D coordinates for pixels in detector with no rotation, shape=(3,N), N=det_size[0]*det_size[1]
	euler_order  : rotation order if you are using euler angles, default="zxz" 
	'''

	if len(rotations) == 0:
		raise RuntimeError("rotations should not be empty !")
	if slice_coor_ori is not None and slice_coor_ori.shape != (3, det_size[0]*det_size[1]):
		raise RuntimeError("input 'slice_coor_ori' shape is invalid !")
	try:
		rotations[0][0]
		squeeze_label = 0
	except:
		rotations = [rotations]
		squeeze_label = 1

	if det_center is None:
		det_center = (np.array(det_size)-1)/2.0
	# make mask
	if mask is not None:
		this_mask = mask.flatten()
		masked_index = np.where(this_mask == 0)[0]
	# make slice
	if slice_coor_ori is None:
		slice_x, slice_y = np.mgrid[0:det_size[0], 0:det_size[1]]
		slice_z = np.zeros(det_size)
		slice_x = slice_x - det_center[0]
		slice_y = slice_y - det_center[1]
		slice_coor_ori = np.vstack([slice_x.flatten(), slice_y.flatten(), slice_z.flatten()])   # shape=(3,N),  N=det_size[0]*det_size[1]		
		del slice_x, slice_y, slice_z
	this_slice = np.zeros( (len(rotations), det_size[0]*det_size[1]), dtype=np.float32 )
	maxR = np.max(np.linalg.norm(slice_coor_ori, axis=0))

	# start slicing
	for ind, rotation in enumerate(rotations):

		# make rotation
		if len(rotation) == 4:
			rot_mat = np.array(quat.quat2rot(rotation))   # shape=(3,3)
		elif len(rotation) == 3:
			rot_mat = np.array(rotate.eul2rotm(rotation, euler_order))  # shape=(3,3)
		else:
			print("get_slice warning: input rotations (%dth) is invalid, skip." % ind)
			continue
		gemm = get_blas_funcs("gemm",[rot_mat, slice_coor_ori])
		slice_coor = gemm(1, rot_mat, slice_coor_ori)
		slice_coor += np.reshape((np.array(model.shape)-1)/2.0, (3,1))  # np.array, shape=(3,N)

		# drop pixels which are out of bound
		drop = 0
		if mask is None:
			if np.ceil(maxR) > np.floor((min(model.shape)-1)/2.0):
				slice_index = np.where( (slice_coor[0]>=0) & (slice_coor[0]<=model.shape[0]-1) & \
										(slice_coor[1]>=0) & (slice_coor[1]<=model.shape[1]-1) & \
										(slice_coor[2]>=0) & (slice_coor[2]<=model.shape[2]-1) )[0]
				slice_coor = slice_coor[:, slice_index]       # shape=(3,N')
				drop = 1
		else:
			if np.ceil(maxR) > np.floor((min(model.shape)-1)/2.0):
				slice_index = np.where( (slice_coor[0]>=0) & (slice_coor[0]<=model.shape[0]-1) & \
										(slice_coor[1]>=0) & (slice_coor[1]<=model.shape[1]-1) & \
										(slice_coor[2]>=0) & (slice_coor[2]<=model.shape[2]-1) & \
										(this_mask==0))[0]
			else:
				slice_index = masked_index
			slice_coor = slice_coor[:, slice_index]       # shape=(3,N')
			drop = 1

		# interpolate
		slice_neighbor = np.zeros(slice_coor.shape, dtype=int)
		weights = np.zeros(slice_coor.shape[1])
		if drop:
			temp_slice = np.zeros(len(slice_index))
		else:
			temp_slice = np.zeros(det_size[0]*det_size[1])
		for i in range(8):
			if i==0:
				slice_neighbor[0] = np.floor(slice_coor[0]).astype(int)
				slice_neighbor[1] = np.ceil(slice_coor[1]).astype(int)
				slice_neighbor[2] = np.ceil(slice_coor[2]).astype(int)
			elif i==1:
				slice_neighbor[0] = np.floor(slice_coor[0]).astype(int)
				slice_neighbor[1] = np.floor(slice_coor[1]).astype(int)
				slice_neighbor[2] = np.ceil(slice_coor[2]).astype(int)
			elif i==2:
				slice_neighbor[0] = np.floor(slice_coor[0]).astype(int)
				slice_neighbor[1] = np.floor(slice_coor[1]).astype(int)
				slice_neighbor[2] = np.floor(slice_coor[2]).astype(int)
			elif i==3:
				slice_neighbor[0] = np.ceil(slice_coor[0]).astype(int)
				slice_neighbor[1] = np.ceil(slice_coor[1]).astype(int)
				slice_neighbor[2] = np.ceil(slice_coor[2]).astype(int)
			elif i==4:
				slice_neighbor[0] = np.ceil(slice_coor[0]).astype(int)
				slice_neighbor[1] = np.floor(slice_coor[1]).astype(int)
				slice_neighbor[2] = np.ceil(slice_coor[2]).astype(int)
			elif i==5:
				slice_neighbor[0] = np.ceil(slice_coor[0]).astype(int)
				slice_neighbor[1] = np.floor(slice_coor[1]).astype(int)
				slice_neighbor[2] = np.floor(slice_coor[2]).astype(int)
			elif i==6:
				slice_neighbor[0] = np.floor(slice_coor[0]).astype(int)
				slice_neighbor[1] = np.ceil(slice_coor[1]).astype(int)
				slice_neighbor[2] = np.floor(slice_coor[2]).astype(int)
			elif i==7:
				slice_neighbor[0] = np.ceil(slice_coor[0]).astype(int)
				slice_neighbor[1] = np.ceil(slice_coor[1]).astype(int)
				slice_neighbor[2] = np.floor(slice_coor[2]).astype(int)
			temp = -np.linalg.norm(slice_neighbor - slice_coor, axis=0)/0.3
			w = ne.evaluate('exp(temp)')
			weights += w
			temp_slice += model[slice_neighbor[0], slice_neighbor[1], slice_neighbor[2]] * w

		if drop:
			this_slice[ind][slice_index] = temp_slice / weights
		else:
			this_slice[ind] = temp_slice / weights

	if squeeze_label == 1:
		return this_slice.reshape((ind+1, det_size[0], det_size[1]))[0]
	else:
		return this_slice.reshape((ind+1, det_size[0], det_size[1]))



def merge_slice(model, rotations, slices, weights=None, det_center=None, \
					mask=None, slice_coor_ori=None, euler_order="zxz"):
	'''
	merge slices into a given model, the orientations depend on quaternions/euler angles
	model : 3d numpy array, original model
	rotations : 2d list, [[w, qx, qy, qz], [alpha, beta, gamma], ...], length = Nq
	            [NOTICE] for euler angle, the default order is intrinsic 'zxz'
	slices : 3d numpy array, some patterns
	weights : same shape with model, initial interpolation weights of all voxels in the model
	det_center : [cx, cy] or None
	mask : np.array, shape=(Nx,Ny), 1 means masked pixel
	slice_coor_ori : well-mapped 3D coordinates for pixels in detector with no rotation, shape=(3,N), N=slices.shape[1]*slices.shape[2]
	euler_order  : rotation order if you are using euler angles, default="zxz" 
	[NOTICE] "model" is modified directly, no return
	'''

	if len(rotations) == 0:
		raise RuntimeError("rotations should not be empty !")
	try:
		rotations[0][0]
	except:
		rotations = [rotations]
	if len(np.array(slices).shape) == 2:
		slices = np.array([slices])

	det_size = (slices.shape[1], slices.shape[2])
	if slice_coor_ori is not None and slice_coor_ori.shape != (3, det_size[0]*det_size[1]):
		raise RuntimeError("input 'slice_coor_ori' shape is invalid !")
	
	if det_center is None:
		det_center = (np.array(det_size)-1)/2.0
	if weights is None:
		weights = np.ones(model.shape, dtype=np.float32)
	# make mask
	if mask is not None:
		this_mask = mask.flatten()
		masked_index = np.where(this_mask == 0)[0]

	# make slice
	if slice_coor_ori is None:
		det_size = slices.shape[1:]
		slice_x, slice_y = np.mgrid[0:det_size[0], 0:det_size[1]]
		slice_z = np.zeros(det_size)
		slice_x = slice_x - det_center[0]
		slice_y = slice_y - det_center[1]
		slice_coor_ori = np.vstack([slice_x.flatten(), slice_y.flatten(), slice_z.flatten()])   # shape=(3,N)
		del slice_x, slice_y, slice_z
	maxR = np.max(np.linalg.norm(slice_coor_ori, axis=0))
	slices_flat = slices.reshape((slices.shape[0], slices.shape[1]*slices.shape[2]))
	
	# start merging
	for ind, rotation in enumerate(rotations):

		# make rotation
		if len(rotation) == 4:
			rot_mat = np.array(quat.quat2rot(rotation))   # shape=(3,3)
		elif len(rotation) == 3:
			rot_mat = np.array(rotate.eul2rotm(rotation, euler_order))  # shape=(3,3)
		else:
			print("get_slice warning: input rotations (%dth) is invalid, skip." % ind)
			continue
		gemm = get_blas_funcs("gemm",[rot_mat, slice_coor_ori])
		slice_coor = gemm(1, rot_mat, slice_coor_ori)
		slice_coor += np.reshape((np.array(model.shape)-1)/2.0, (3,1))  # np.array, shape=(3,N)

		# drop pixels which are out of bound
		if mask is None:
			if np.ceil(maxR) > np.floor((min(model.shape)-1)/2.0):
				slice_index = np.where( (slice_coor[0]>=0) & (slice_coor[0]<=model.shape[0]-1) & \
										(slice_coor[1]>=0) & (slice_coor[1]<=model.shape[1]-1) & \
										(slice_coor[2]>=0) & (slice_coor[2]<=model.shape[2]-1) )[0]
				slice_coor = slice_coor[:, slice_index]       # shape=(3,N')
				this_slice_flat = slices_flat[ind][slice_index]  # shape=(N',)
			else:
				this_slice_flat = slices_flat[ind].flatten()
		else:
			if np.ceil(maxR) > np.floor((min(model.shape)-1)/2.0):
				slice_index = np.where( (slice_coor[0]>=0) & (slice_coor[0]<=model.shape[0]-1) & \
										(slice_coor[1]>=0) & (slice_coor[1]<=model.shape[1]-1) & \
										(slice_coor[2]>=0) & (slice_coor[2]<=model.shape[2]-1) & \
										(this_mask==0))[0]
			else:
				slice_index = masked_index
			slice_coor = slice_coor[:, slice_index]       # shape=(3,N')
			this_slice_flat = slices_flat[ind][slice_index]  # shape=(N',)

		# interpolate
		slice_neighbor = np.zeros(slice_coor.shape, dtype=int)
		for i in range(8):
			if i==0:
				slice_neighbor[0] = np.floor(slice_coor[0]).astype(int)
				slice_neighbor[1] = np.ceil(slice_coor[1]).astype(int)
				slice_neighbor[2] = np.ceil(slice_coor[2]).astype(int)
			elif i==1:
				slice_neighbor[0] = np.floor(slice_coor[0]).astype(int)
				slice_neighbor[1] = np.floor(slice_coor[1]).astype(int)
				slice_neighbor[2] = np.ceil(slice_coor[2]).astype(int)
			elif i==2:
				slice_neighbor[0] = np.floor(slice_coor[0]).astype(int)
				slice_neighbor[1] = np.floor(slice_coor[1]).astype(int)
				slice_neighbor[2] = np.floor(slice_coor[2]).astype(int)
			elif i==3:
				slice_neighbor[0] = np.ceil(slice_coor[0]).astype(int)
				slice_neighbor[1] = np.ceil(slice_coor[1]).astype(int)
				slice_neighbor[2] = np.ceil(slice_coor[2]).astype(int)
			elif i==4:
				slice_neighbor[0] = np.ceil(slice_coor[0]).astype(int)
				slice_neighbor[1] = np.floor(slice_coor[1]).astype(int)
				slice_neighbor[2] = np.ceil(slice_coor[2]).astype(int)
			elif i==5:
				slice_neighbor[0] = np.ceil(slice_coor[0]).astype(int)
				slice_neighbor[1] = np.floor(slice_coor[1]).astype(int)
				slice_neighbor[2] = np.floor(slice_coor[2]).astype(int)
			elif i==6:
				slice_neighbor[0] = np.floor(slice_coor[0]).astype(int)
				slice_neighbor[1] = np.ceil(slice_coor[1]).astype(int)
				slice_neighbor[2] = np.floor(slice_coor[2]).astype(int)
			elif i==7:
				slice_neighbor[0] = np.ceil(slice_coor[0]).astype(int)
				slice_neighbor[1] = np.ceil(slice_coor[1]).astype(int)
				slice_neighbor[2] = np.floor(slice_coor[2]).astype(int)

			temp = -np.linalg.norm(slice_neighbor - slice_coor, axis=0)
			w = ne.evaluate("exp(temp/0.3)")
			
			temp_weight = weights[slice_neighbor[0], slice_neighbor[1], slice_neighbor[2]]
			temp_model = model[slice_neighbor[0], slice_neighbor[1], slice_neighbor[2]]
			temp_model = ne.evaluate('temp_model * temp_weight + this_slice_flat * w')
			temp_weight = temp_weight + w
			temp_model = ne.evaluate('temp_model / temp_weight')
			weights[slice_neighbor[0], slice_neighbor[1], slice_neighbor[2]] = temp_weight
			model[slice_neighbor[0], slice_neighbor[1], slice_neighbor[2]] = temp_model



def poisson_likelihood(W_j, K_k, beta=1, weight=None):
	'''
	calculate poisson likelihood R_jk between model slice (W_j) and experimental pattern (K_k)

	Do masking ahead of using this function.
	Final return is the value weight*(R_jk^beta)
	'''

	W_j_mask_index = np.where(W_j>0)
	W_j_ = W_j[W_j_mask_index]
	K_k_ = K_k[W_j_mask_index]
	temp = ne.evaluate('sum(K_k_*log(W_j_)-W_j_)')/np.product(W_j_.shape)
	if weight is not None:
		R_jk = ne.evaluate('exp(temp*beta)*weight')
	else:
		R_jk = ne.evaluate('exp(temp*beta)')

	return float(R_jk)



def maximization(K_ks, Prob_ks):
	'''
	Calculate updated tomograph of one orientation and return

	Input : K_ks , all useful patterns, please reshape pattern to array or do masking in advance
			Prob_ks , probabilities of all useful patterns (after normalizing in every orientation) in orientation j
	'''

	assert len(K_ks.shape) == 2, "please reshape pattern to array or do masking in advance"

	Prob_norm = (Prob_ks/sum(Prob_ks)).reshape((len(Prob_ks),1))
	W_j_prime = ne.evaluate('sum(K_ks * Prob_norm, axis=0)')

	return W_j_prime



def get_quaternion(Num_level):
	'''
	Generate quaternions as uniform sampling in rotation space (weights = 1), based on Fibonacci spherical sampling

	Input : Num_level, a integer controlling the number of quaternions
			The number of output quaternions is 2*Num_level^3
	'''
	from spipy.analyse import orientation
	from spipy.image import quat

	assert Num_level>0, "Num_level should be >0"

	num_vec = 2*Num_level**2
	vec_n,_ = orientation.Sphere_randp('uniform-1', 1, num_vec)
	inp_r = np.linspace(0, np.pi*2, Num_level+1)[:Num_level]
	inp_r = inp_r.reshape((len(inp_r),1))
	quaternions = np.zeros((2*Num_level**3,4))

	ind = 0
	for vec in vec_n:
		for inp in inp_r:
			quaternions[ind] = quat.azi2quat([inp,vec[0],vec[1],vec[2]])
			ind += 1
	return quaternions

