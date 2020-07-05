import os
import sys
import numpy as np
import configparser
import subprocess
import h5py

_workpath = None
config_essential = {'parameters|detd' : 200, 'parameters|lambda' : 2.5, \
					'parameters|detsize' : '128 128', 'parameters|pixsize' : 0.3, \
					'parameters|stoprad' : 40, 'parameters|polarization' : 'x', \
					'emc|num_div' : 10, 'emc|need_scaling' : 1, \
					'emc|beta' : 0.006, 'emc|beta_schedule' : '1.414 10' }
config_advanced = {'parameters|ewald_rad' : 'None', 'make_detector|in_mask_file' : 'None', \
					'emc|sym_icosahedral' : 0, 'emc|selection' : 'None', \
					'emc|start_model_file' : 'None'}

def help(module):
	if module=="use_project":
		print("This function is used to switch to a existing project")
		print("    -> Input: project_path ( str, the project directory you want to switch to)")
		return
	elif module=="new_project":
		print("This function is used to create a new project directory at your given path")
		print("    -> Input: data_path (path of your patterns' file, MUST be h5 file)")
		print("              inh5 (path of patterns inside h5 file, patterns should be stored in a numpy.ndarray, shape=(Nd,Nx,Ny))")
		print("     *option: path (create work directory at your give path, default as current dir)")
		print("     *option: name (give a name to your project, default is an number)")
		print("[Notice] 'path' must be absolute path !")
		return
	elif module=="config":
		print("This function is used to edit configure file")
		print("    -> Input (dict, parameters yout want to modified.)")
		print("params format : ")
		print("    {\n\
					'parameters|detd' : 200, \n\
					'emc|num_div' : 10, \n\
					... \n\
					}")
		print("You can look into 'Config' part of README document for detail information;")
		print("or refer to 'emc.config_essential' attribute for default values of neccessary parameters, ")
		print("and 'emc.config_advanced' attribute for default values of optional parameters.")
		print("Help exit.")
		return
	elif module=="run":
		print("Call this function to start emc")
		print("    -> Input: num_proc (int, how many processes to run in parallel)")
		print("              num_thread (int, how many threads in each process) ")
		print("              iters (int, how many reconstruction iterations)")
		print("       *option: nohup (bool, whether run in the background, default=False)")
		print("       *option: resume (bool, whether run from previous break point, default=False)")
		print("       *option: cluster (bool, whether you will submit jobs using job scheduling system, if yes, the function will only generate a command file at your work path without submitting it, and ignore nohup value. default=True)")
		print("    -> Return: submit command, only if cluster == True")
		print("[Notice] As this program costs a lot of memories, use as less processes and much threads as possible.\
			Recommended strategy : num_proc * num_thread ~ number of cores in your CPUs. Let one cluster node support 1~2 processes. (Mentioned, large processes number may cause low precision in merging result)")
		return
	else:
		raise ValueError("No module names "+str(module))

def use_project(project_path):
	global _workpath

	temp = None
	if project_path[0] == '/' or project_path[0:2] == '~/':
		temp = os.path.abspath(project_path)
		if os.path.exists(temp):
			_workpath = temp
		else:
			raise ValueError("The project " + temp + " doesn't exists. Exit")
	else:
		nowfolder = os.path.abspath(sys.path[0])
		temp = os.path.join(nowfolder, project_path)
		if os.path.exists(temp):
			_workpath = os.path.abspath(temp)
		else:
			raise ValueError("The project " + temp + " doesn't exists. Exit")

def new_project(data_path, inh5, path=None, name=None):
	global _workpath

	code_path = __file__.split('/emc.py')[0] + '/template_emc'
	if not os.path.exists(data_path):
		raise ValueError("Your data path is incorrect. Try ABSOLUTE PATH. Exit\n")
	else:
		if inh5 is None:
			raise ValueError("Give me patterns' path inside h5 file\n")
		else:
			try:
				tempf = h5py.File(data_path, 'r')
				temp = tempf[inh5][0]
				tempf.close()
				temp.shape[0]
				temp.shape[1]
			except:
				raise RuntimeError("Check your input data file. Some errors arise when I try to load data.\n")
	if path == None or path == "./":
		path = os.path.abspath(sys.path[0])
	else:
		if not os.path.exists(path):
			raise ValueError('Your path is incorrect. Try ABSOLUTE PATH. Exit\n')
		else:
			path = os.path.abspath(path)
	if name is not None:
		_workpath = os.path.join(path, name)
	else:
		all_dirs = os.listdir(path)
		nid = 0
		for di in all_dirs:
			if di[0:4] == "emc_" and str.isdigit(di[4:]):
				nid = max(nid, int(di[4:]))
		nid += 1
		_workpath = os.path.join(path, 'emc_' + format(nid, '02d'))
	cmd = code_path + '/new_project ' + _workpath
	subprocess.check_call(cmd, shell=True)
	# now change output|path in config.ini
	config = configparser.ConfigParser()
	config.read(os.path.join(_workpath, 'config.ini'))
	config.set('make_detector', 'out_detector_file', os.path.join(_workpath, 'data/det_exp.dat'))
	config.set('make_data', 'out_photons_file', os.path.join(_workpath, 'data/photons.emc'))
	config.set('emc', 'output_folder', os.path.join(_workpath, 'data/'))
	config.set('emc', 'log_file', os.path.join(_workpath, 'EMC.log'))
	with open(os.path.join(_workpath, 'config.ini'), 'w') as f:
		config.write(f)
	# now compile emc and link it to workpath
	cmd = os.path.join(code_path, 'src/compile.sh') + ' ' + _workpath
	subprocess.check_call(cmd, shell=True)
	# now transfer h5 file to emc data and save to _workpath
	cmd = 'python ' + os.path.join(code_path, 'py_src/make_emc.py') + ' ' \
			+ data_path + ' ' + inh5 + ' ' + os.path.join(_workpath, 'data/photons.emc')
	subprocess.check_call(cmd, shell=True)
	print("\nAll work done ! ")
	print("Now please confirm running parameters. Your can re-edit it by calling function emc.config(...) or eidt config.ini directly.\n")
	print("Notice : DO NOT change any paths in config.ini !!!")

def config(params):
	global _workpath

	if not os.path.exists(os.path.join(_workpath,'config.ini')):
		raise RuntimeError("I can't find your configure file, please run emc.new_project(...) first !")
	if type(params)!=dict:
		raise ValueError("Your input 'params' is invalid. Exit")

	if params == {}:
		pass
	else:
		maskpath = params['make_detector|in_mask_file']
		if maskpath is None:
			params['make_detector|in_mask_file'] = None
		elif not os.path.exists(maskpath):
			raise RuntimeError("I can't find your mask file. Check it please.")
		else:
			if 'byt' in os.path.splitext(maskpath)[-1] or 'bin' in os.path.splitext(maskpath)[-1]:
				cmd = "cp " + os.path.abspath(maskpath) + " " + os.path.join(_workpath, "mask.byt")
				subprocess.check_call(cmd, shell=True)
			elif 'npy' in os.path.splitext(maskpath)[-1]:
				masktmp = np.load(maskpath)
				masktmp = masktmp.astype("uint8")
				masktmp.tofile(os.path.join(_workpath, "mask.byt"))
			else:
				raise RuntimeError("I only support .byt, .bin or .npy format !")
			params['make_detector|in_mask_file'] = os.path.join(_workpath, "mask.byt")
		config = configparser.ConfigParser()
		config.read(os.path.join(_workpath,'config.ini'))
		for k in params.keys():
			section, par = k.split("|")
			config.set(section, par, str(params[k]))
		with open(os.path.join(_workpath,'config.ini'), 'w') as f:
			config.write(f)
	# write data/detector.dat
	code_path = __file__.split('/emc.py')[0] + '/template_emc'
	cmd = "python " + os.path.join(code_path,'make_detector.py') + ' -c '+ os.path.join(_workpath,'config.ini')
	subprocess.check_call(cmd, shell=True)
	print('\n Configure finished.')

def run(num_proc, num_thread, iters, nohup=False, resume=False, cluster=True):
	global _workpath

	num_proc = int(num_proc)
	if num_thread is None or iters is None:
		raise RuntimeError("I can't find input 'num_thread' or 'iters'")
	else:
		num_thread = int(abs(num_thread))
		iters = int(abs(iters))
	
	if not os.path.exists(os.path.join(_workpath,'data/det_exp.dat')):
		raise ValueError("Please call emc.new_project(...) and emc.config(...) first ! Exit")

	code_path = __file__.split('/emc.py')[0] + '/template_emc'

	if cluster:
		from .. import info

		try:
			mpirun = info.EMC_MPI
		except:
			mpirun = "mpiexec"

		print("\n Make directories, check submit_job.sh for details.\n")
		cmd = mpirun + ' -np ' + str(num_proc) + ' ./emc -c config.ini' + ' -t ' +  str(num_thread)
		# check resume and nohup
		if resume:
			cmd = cmd + ' -r ' + str(iters)
		else:
			cmd = cmd + ' ' + str(iters)
		submitfile = open(os.path.join(_workpath, "submit_job.sh"), 'w')
		submitfile.write("#! /bin/bash\n\n")
		submitfile.write("# COMMAND EXPLANATION:\n")
		submitfile.write("# mpiexec -np <p> ./emc -c config.ini -t <t> -r <i>\n")
		submitfile.write("#     -np <p> number of processes\n")
		submitfile.write("#     -t <t> number of threads per process\n")
		submitfile.write("#     -r resume from previous breakpoint\n")
		submitfile.write("#     <i> number of iterations\n")
		submitfile.write("\n# Submit the command below to your job submitting system to run emc\n")
		submitfile.write(cmd + '\n')
		submitfile.close()
		return cmd

	if not cluster:

		if not os.path.isfile(os.path.join(_workpath, 'emc')):
			print("\n EMC module is not compiled, exit.")
			return

		cmd = os.path.join(_workpath, 'emc') + ' -c ' + os.path.join(_workpath, 'config.ini') + ' -t ' +  str(num_thread)
		if resume:
			cmd = cmd + ' -r ' + str(iters)
		else:
			cmd = cmd + ' ' + str(iters)
		if nohup:
			cmd = cmd + ' &>' + os.path.join(_workpath, "emc_details.log") + '&'
		subprocess.check_call(cmd, shell=True)
