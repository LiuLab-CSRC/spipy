import numpy as np
from scipy.ndimage import interpolation
from . import criterion
import copy

def help(module):
	if module=="eul2rotm":
		print("Use this function to transfer euler angles to rotation matrix")
		print("    -> Input: ea (euler angle in rad, list/array, intrinsic)")
		print("              order (rotation order, str, such as 'zxz')")
		print("    -> Output: 3x3 rotation matrix, numpy.ndarray")
		return
	elif module=="rot_ext":
		print("This function is used to do an extrinsic rotation to a 3D voxel model")
		print("    -> Input: ea (euler angle in rad, list/array, extrinsic)")
		print("              order (rotation order, str, such as 'zyz')")
		print("              matrix (numpy.ndarray, 3D matrix that to be rotated)")
		print("      option: ref (reference voxel model in 3d matrix, if given, program will compare rotated matrix\
		 with reference by ploting their x/y/z slices. default is None)")
		print("    -> Output: d (rotated model)")
		return
	elif module=="align":
		print("Using euler angle grid search to align two models, fix and mov")
		print("    -> Input: fix (fixed model, 3d numpy.ndarray)")
		print("              mov (moving model, 3d numpy.ndarray, should have the same size with fix)")
		print("      option: grid_unit (search grid of euler angles, list, default=[0.3,0.1],\
		 which means the program will firstly search with grid=0.3 rad, then fine-tune with grid=0.1 rad)")
		print("      option: nproc (number of processes running in parallel, parallel on alpha angle, default=2)")
		print("      option: resize (int, resize original matrix to smaller size, default=40 pixels)")
		print("      option: order (rotation order, str, default='zxz')")
		print("    -> Output: r_factor (the overall r-factor between fixed and best aligned mov model)")
		print("               ea (best aligned euler angle)")
		print("               new_mov (mov model after aligned)")
		return
	else:
		raise ValueError("No module names "+str(module))

def eul2rotm(ea, order):

	def Ry(e):
		return np.array([[np.cos(e), 0, -np.sin(e)],\
				[0, 1, 0],\
				[np.sin(e), 0, np.cos(e)]])
	def Rz(e):
		return np.array([[np.cos(e), np.sin(e), 0],\
				[-np.sin(e), np.cos(e), 0],\
				[0, 0, 1]])
	def Rx(e):
		return np.array([[1, 0, 0],\
				[0, np.cos(e), np.sin(e)],\
				[0, -np.sin(e), np.cos(e)]])

	T = np.diag([1,1,1])
	for ind,o in enumerate(order):
		if o=='z':
			T = np.dot( T, Rz(ea[ind]) )
		elif o=='y':
			T = np.dot( T, Ry(ea[ind]) )
		elif o=='x':
			T = np.dot( T, Rx(ea[ind]) )
	return T

def rot_ext(ea, order, vol, ref=None):
	if len(vol.shape)!=3:
		raise RuntimeError("Input matrix should be 3-dimension!")
	angle = np.array(ea)/np.pi*180
	d = copy.deepcopy(vol)
	for ind,o in enumerate(order):
		if o=='z':
			d = interpolation.rotate(d,angle[ind],(0,1),False)
		elif o=='y':
			d = interpolation.rotate(d,angle[ind],(0,2),False)
		elif o=='x':
			d = interpolation.rotate(d,angle[ind],(1,2),False)
	if ref is not None:
		import matplotlib.pyplot as plt
		size = list(vol.shape)
		plt.figure(figsize=(14,7))
		plt.subplot(2,3,1)
		plt.imshow(np.log(1+np.abs(ref[size[0]//2,:,:])))
		plt.title('xPlain-ref')
		plt.subplot(2,3,2)
		plt.imshow(np.log(1+np.abs(ref[:,size[1]//2,:])))
		plt.title('yPlain-ref')
		plt.subplot(2,3,3)
		plt.imshow(np.log(1+np.abs(ref[:,:,size[2]//2])))
		plt.title('zPlain-ref')
		plt.subplot(2,3,4)
		plt.imshow(np.log(1+np.abs(d[size[0]//2,:,:])))
		plt.title('xPlain-newd')
		plt.subplot(2,3,5)
		plt.imshow(np.log(1+np.abs(d[:,size[1]//2,:])))
		plt.title('yPlain-newd')
		plt.subplot(2,3,6)
		plt.imshow(np.log(1+np.abs(d[:,:,size[2]//2])))
		plt.title('zPlain-newd')
		plt.show()
	return d

def _grid_search(grid_alpha, grid_beta, grid_gamma, fix, mov, order):
	best_r_factor = 1.0
	best_ea = None
	for alpha in grid_alpha:
		for beta in grid_beta:
			for gamma in grid_gamma:
				moved = rot_ext([alpha, beta, gamma], order, mov)
				rfactor = criterion.r_factor(fix, moved)
				if rfactor < best_r_factor:
					best_r_factor = rfactor
					best_ea = [alpha, beta, gamma]
					print("..best_rf = " + str(best_r_factor) + "..best_ea = " + str(best_ea))
	return [best_r_factor, best_ea]

def align(fix, mov, grid_unit=[0.3, 0.1], nproc=2, resize=40, order='zxz'):
	import multiprocessing as mp
	if fix.shape != mov.shape:
		raise RuntimeError("Input fix and mov should has the same shape!")
	print("\n************INPUT*****************")
	print("grid unit list : " + str(grid_unit))
	print("order : " + order)
	print("number of processes : " + str(nproc))
	print("************START*****************")
	# preprocess
	zoom_ratio = float(resize)/fix.shape[0]
	newfix = interpolation.zoom(fix, zoom_ratio)
	newmov = interpolation.zoom(mov, zoom_ratio)
	scaling_f = np.linalg.norm(newfix)/np.linalg.norm(newmov)
	newmov *= scaling_f
	# alingment
	global_rf = 1.0
	global_ea = None
	grid_unit = sorted(list(set(grid_unit)), reverse=True)
	for gi,gu in enumerate(grid_unit):
		print("\n================Calculating, grid unit = " + str(gu))
		if gi==0:
			grid_alpha = np.linspace(0, np.pi*2, np.pi*2/gu)
			grid_beta = np.linspace(0, np.pi, np.pi*2/gu)
			grid_gamma = np.linspace(0, np.pi*2, np.pi*2/gu)
		else:
			grid_alpha = np.linspace(global_ea[0] - grid_unit[gi-1],\
						 global_ea[0] + grid_unit[gi-1],\
						 grid_unit[gi-1]*2/gu)
			grid_beta = np.linspace(global_ea[1] - grid_unit[gi-1],\
						 global_ea[1] + grid_unit[gi-1],\
						 grid_unit[gi-1]*2/gu)
			grid_gamma = np.linspace(global_ea[2] - grid_unit[gi-1],\
						 global_ea[2] + grid_unit[gi-1],\
						 grid_unit[gi-1]*2/gu)
		# start parallel
		result = []
		partition = np.linspace(0, len(grid_alpha), nproc+1, dtype=int)
		pool = mp.Pool(processes = nproc)
		for i in np.arange(nproc):
			print("Start process " + str(i) + " ...")
			alpha_grid_part = grid_alpha[partition[i]:partition[i+1]]
			result.append( pool.apply_async(_grid_search, args=(alpha_grid_part,grid_beta,grid_gamma,\
							newfix,newmov,order,)) )
		pool.close()
		pool.join()
		for ind,p in enumerate(result):
			re = p.get()
			if re[0] < global_rf:
				global_rf = re[0]
				global_ea = re[1]
		print("==================End. best_r_factor = " + str(global_rf) + "best_ea = " + str(global_ea))
	new_mov = rot_ext(global_ea, order, mov)
	return global_rf, global_ea, new_mov


