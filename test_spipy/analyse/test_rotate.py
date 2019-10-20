import sys
from spipy.analyse import rotate
from spipy.analyse import criterion
import numpy as np
import h5py
import os
import copy

if __name__=="__main__":
	'''
	align and compare two models
	python test_rotate.py [path_fixed_model] [path_moving model] [angle grid] [number of processes] [output dir]
	'''
	try:
		fix = 'volume.bin'
		mov = 'volume_rotated.bin'
		grid_unit = map(np.float,sys.argv[1].split(','))
		nproc = int(sys.argv[2])
		save_dir = sys.argv[3]
	except:
		print("python test_rotate.py [angle grid] [number of processes] [output dir]")
		sys.exit(0)
	
	if not os.path.isdir(save_dir):
		raise RuntimeError("Output path is invalid!")
	
	outer_cut = 50
	inner_cut = 10

	d1 = np.fromfile(fix).reshape((125,125,125))
	d2 = np.fromfile(mov).reshape((125,125,125))
	center = np.array(d1.shape, dtype=int)//2
	d1_small = copy.deepcopy(d1[center[0]-outer_cut:center[0]+outer_cut, center[1]-outer_cut:center[1]+outer_cut, center[2]-outer_cut:center[2]+outer_cut])
	d2_small = copy.deepcopy(d2[center[0]-outer_cut:center[0]+outer_cut, center[1]-outer_cut:center[1]+outer_cut, center[2]-outer_cut:center[2]+outer_cut])
	d1_small[center[0]-inner_cut:center[0]+inner_cut, center[1]-inner_cut:center[1]+inner_cut, center[2]-inner_cut:center[2]+inner_cut] = 0
	d2_small[center[0]-inner_cut:center[0]+inner_cut, center[1]-inner_cut:center[1]+inner_cut, center[2]-inner_cut:center[2]+inner_cut] = 0

	_,ea,_ = rotate.align(d1_small, d2_small, grid_unit, nproc)
	newd2 = rotate.rot_ext(ea, 'zxz', d2)
	rf = criterion.r_factor_shell(newd2, d1, np.arange(2, d2.shape[0]//2-1))
	rf_all = criterion.r_factor(newd2, d1)
	# calculate reverse
	rev_ea = -np.array(ea[::-1])
	rev_d2 = rotate.rot_ext(rev_ea, 'zxz', newd2)
	rev_rf = criterion.r_factor(rev_d2, d2)
	rev_rf_shell = criterion.r_factor_shell(rev_d2, d2, np.arange(d2.shape[0]//2-1))
	# save
	saved = h5py.File(os.path.join(save_dir,'compare.h5'),'w')
	saved.create_dataset('fix-mov',data='fix : '+str(fix)+', mov : '+str(mov))
	saved.create_dataset('rot_order',data='extrinsic zxz')
	saved.create_dataset('best_euler',data=ea)
	saved.create_dataset('r_factor',data=rf)	
	saved.create_dataset('mov_after_rotate',data=newd2)
	saved.create_dataset('reverse_r_factor',data=rev_rf_shell)
	saved.close()
	# print
	rotm = rotate.eul2rotm(ea, 'zxz')
	print("\nd : euler angle = " + str(ea))
	print("d : rotation matrix = " + str(rotm))
	print("d : over-all r factor = " + str(rf_all))
	print("d : over-all reverse r factor = " + str(rev_rf))
	print("Saving to files : "+os.path.join(save_dir,'compare.h5')+"\n")
