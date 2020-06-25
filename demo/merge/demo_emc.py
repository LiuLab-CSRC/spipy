import numpy as np
import h5py
import sys
from spipy.merge import emc

if __name__=="__main__":
	
	print("\n(1) test emc.new_project ..")
	fpath = '../test_pattern.h5'
	emc.new_project(data_path = fpath,\
			inh5 = "patterns",\
			path = "./",\
			name = None)
	

	print("\n(2) test emc.config ..")
	config_essential = {'parameters|detd' : 581, 'parameters|lambda' : 7.9, \
					'parameters|detsize' : '128 128', 'parameters|pixsize' : 0.6, \
					'parameters|stoprad' : 20, 'parameters|polarization' : 'x', \
					'emc|num_div' : 10, 'emc|need_scaling' : 1, \
					'emc|beta' : 0.006, 'emc|beta_schedule' : '1.414 10' }
	config_optional = {'parameters|ewald_rad' : '650.', 'make_detector|in_mask_file' : 'sample_mask_small.byt', \
					'emc|sym_icosahedral' : 0, 'emc|selection' : 'None', \
					'emc|start_model_file' : 'None'}
	params = dict(config_essential, **config_optional)
	emc.config(params)


	print("\n(3) test emc.run ..")
	emc.run(num_proc=8, num_thread=12, iters=5, nohup=False, resume=False, cluster=True)
	sys.exit(0)
	
	
	# use 4 CPUs with 24 cores, and 5 iterations (this is just a test, usually iter number >= 30)
	# Put this onto a cluster !!!
	emc.use_project('emc_01')
	emc.run(num_proc=8, num_thread=12, iters=5, nohup=False, resume=False, cluster=False)
