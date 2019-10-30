from spipy.phase import phase3d
import sys

if __name__=="__main__":

	params_essential = {'input|shape' : '125,125,125', 'input|padd_to_pow2' : True, \
		'input|inner_mask' : 4, 'input|outer_mask' : 55, \
		'input|outer_outer_mask' : 64, 'input|mask_edges' : True, \
		'phasing|repeats' : 1, 'phasing|iters' : '100RAAR 100DM 200ERA', \
		'phasing_parameters|voxel_number' : 2000, 'phasing_parameters|beta' : 0.8}
	params_optional = {'input|subtract_percentile' : None, 'input|spherical_support' : None, \
		'phasing_parameters|background' : 'True', 'input|init_model' : None}

	print("\nCreate new project ...")
	phase3d.new_project(data_path='3dvolume.bin', mask_path=None, path='./', name='mytest')

	print("\nConfiguring ...")
	parameters = dict(params_essential, **params_optional)
	phase3d.config_project(params = parameters)

	print("Start !")
	phase3d.run_project(num_proc=2,nohup=False,cluster=False)

	sys.exit(0)
	phase3d.use_project('./mytest')
	phase3d.show_result(outpath=None, exp_param='581,7.9,128,0.3')
