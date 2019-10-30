import os
import sys
import numpy as np
import configparser
import subprocess

__workpath = None

def help(module):
	if module=="use_project":
		print("This function is used to switch to a existing project")
		print("    -> Input: project_path ( str, the project directory you want to switch to)")
		return
	elif module=="new_project":
		print("This function is used to create a new project directory at your given path")
		print("    -> Input: data_path (path of your original intensity data)")
		print("     *option: mask_path (path of your mask file (.npy file), default=None)")
		print("     *option: path (create work directory at your give path, default as current dir)")
		print("     *option: name (give a name to your project, default is an number)")
		print("[Notice] Your original intensity file should be 3D matrix '.npy' or '.mat', or Dragonfly output '.bin'")
		print("[Notice] 'path' must be absolute path !")
		return
	elif module=="config_project":
		print("This function is used to edit configure file")
		print("    -> Input (dict, parameters yout want to modified.)")
		print("params format : ")
		print("    {\n\
					'input|shape' : '120,120,120', \n\
					'input|padd_to_pow2' : 'True', \n\
					... \n\
					}")
		print("You can look into 'config.ini' for detail information")
		return
	elif module=="run_project":
		print("Call this function to start phasing")
		print("    -> *option: num_proc (int, how many processes to run in parallel, default=1)")
		print("       *option: nohup (bool, whether run it in the background, default=False)")
		print("       *option: cluster (bool, whether you will submit jobs using job scheduling system, if yes, the function will only generate a command file at your work path without submitting it, and ignore nohup value. default=True))")
		return
	else:
		raise ValueError("No module names "+str(module))

def use_project(project_path):
	global __workpath
	temp = None
	if project_path[0] == '/' or project_path[0:2] == '~/':
		temp = os.path.abspath(project_path)
		if os.path.exists(temp):
			__workpath = temp
		else:
			raise ValueError("The project " + temp + " doesn't exists. Exit")
	else:
		nowfolder = os.path.abspath(sys.path[0])
		temp = os.path.join(nowfolder, project_path)
		if os.path.exists(temp):
			__workpath = os.path.abspath(temp)
		else:
			raise ValueError("The project " + temp + " doesn't exists. Exit")

def new_project(data_path, mask_path=None, path=None, name=None):
	global __workpath

	code_path = __file__.split('/phase3d.py')[0]
	if not os.path.exists(data_path):
		raise ValueError("\nYour data path is incorrect. Try ABSOLUTE PATH. Exit\n")
	if mask_path is not None and not os.path.exists(mask_path):
		raise ValueError("\nYour mask path is incorrect. Try ABSOLUTE PATH. Exit\n")
	if path == None or path == "./":
		path = os.path.abspath(sys.path[0])
	else:
		if not os.path.exists(path):
			raise ValueError('\n Your path is incorrect. Try ABSOLUTE PATH. Exit\n')
		else:
			path = os.path.abspath(path)
	if name is not None:
		__workpath = os.path.join(path, name)
	else:
		all_dirs = os.listdir(path)
		nid = 0
		for di in all_dirs:
			if di[0:8] == "phase3d_" and str.isdigit(di[8:]):
				nid = max(nid, int(di[8:]))
		nid += 1
		__workpath = os.path.join(path, 'phase3d_' + format(nid, '02d'))
	cmd = code_path + '/template_3d/new_project ' + __workpath
	subprocess.check_call(cmd, shell=True)
	# now change output|path in config.ini
	config = configparser.ConfigParser()
	config.read(os.path.join(__workpath, 'config.ini'))
	config.set('output', 'path', __workpath)
	config.set('input', 'fnam', os.path.join(__workpath,'data.bin'))
	# now load data
	if data_path.split('.')[-1] == 'npy':
		data = np.load(data_path)
		data.tofile(__workpath+'/ori_intens/intensity.bin')
		config.set('input', 'dtype', str(data.dtype))
	elif data_path.split('.')[-1] == 'bin':
		cmd = 'cp ' + data_path + ' ' + __workpath + '/ori_intens/intensity.bin'
		subprocess.check_call(cmd, shell=True)
	elif data_path.split('.')[-1] == 'mat':
		import scipy.io as sio
		dfile = sio.loadmat(data_path)
		data = dfile.values()[0]
		data.tofile(__workpath+'/ori_intens/intensity.bin')
		config.set('input', 'dtype', str(data.dtype))
	else:
		raise ValueError('\n Error while loading your data ! Exit\n')
	# now load mask
	if mask_path is not None:
		cmd = 'cp ' + mask_path + ' ' + __workpath + '/ori_intens/mask.npy'
		subprocess.check_call(cmd, shell=True)
		cmd = 'ln -fs ' + __workpath + '/ori_intens/mask.npy ' + __workpath + '/mask.npy'
		subprocess.check_call(cmd, shell=True)
	# now write config.ini
	with open(os.path.join(__workpath, 'config.ini'), 'w') as f:
		config.write(f)
	cmd = 'ln -fs ' + __workpath + '/ori_intens/intensity.bin ' + __workpath + '/data.bin'
	subprocess.check_call(cmd, shell=True)
	print("\nAll work done ! ")
	print("Now please confirm running parameters. Your can re-edit it by calling function phase3d.config(...) or eidt config.ini directly.\n")

def config_project(params):
	global __workpath
	from .template_3d import make_input

	if not os.path.exists(os.path.join(__workpath,'config.ini')):
		raise ValueError("I can't find your configure file, please run phase3d.new_project(...) first !")

	config = configparser.ConfigParser()
	config.read(os.path.join(__workpath,'config.ini'))
	for k in params.keys():
		section, par = k.split("|")
		config.set(section, par, str(params[k]))
	with open(os.path.join(__workpath,'config.ini'), 'w') as f:
		config.write(f)

	# make input
	make_input.make_input(os.path.join(__workpath,'config.ini'))
	print('\n Configure finished.')


def run_project(num_proc=1, nohup=False, cluster=True):
	global __workpath

	if not os.path.exists(os.path.join(__workpath,'input.h5')):
		raise ValueError("Please call phase3d.new_project(...) and phase3d.config(...) first ! Exit")

	code_path = __file__.split('/phase3d.py')[0] + '/template_3d'

	if num_proc >= 1:
		# python path
		pythony = subprocess.check_output('which python', shell=True).decode().strip("\n")
		# mpirun path
		mpirun = pythony.split('bin')[0]
		mpirun = os.path.join(mpirun, 'bin/mpiexec')
		if nohup == True:
			cmd = mpirun + " -np "+str(num_proc)+" --mca btl self,tcp --oversubscribe %s " % pythony \
				+ os.path.join(code_path, 'phase.py') + ' ' + os.path.join(__workpath, 'input.h5') + ' &>' + os.path.join(__workpath, 'phase.log') + '&'
		else:
			cmd = mpirun + " -np "+str(num_proc)+" --mca btl self,tcp --oversubscribe %s " % pythony \
				+ os.path.join(code_path, 'phase.py') + ' ' + os.path.join(__workpath, 'input.h5')
		if cluster:
			print("\n Dry run on cluster, check submit_job.sh for details.\n")
			submitfile = open(os.path.join(__workpath, "submit_job.sh"), 'w')
			submitfile.write("# Submit the command below to your job submitting system to run 3d phasing\n")
			submitfile.write(cmd + '\n')
			submitfile.close()
		else:
			subprocess.check_call(cmd, shell=True)
	else:
		raise RuntimeError('num_proc should be a positive integer ! Exit.')

'''
def show_result(outpath=None, exp_param=None):
	global __workpath
	if outpath is not None and type(outpath)!=str:
		raise ValueError("Input 'outpath should be a string. Exit'")

	code_path = __file__.split('/phase3d.py')[0] + '/template_3d'

	if outpath is None:
		cmd = "python " + os.path.join(code_path, 'show_result.py') + ' ' + os.path.join(__workpath, 'output.h5')
	else:
		cmd = "python " + os.path.join(code_path, 'show_result.py') + ' ' + outpath
	if exp_param is not None:
		cmd = cmd + ' ' + str(exp_param)

	subprocess.check_call(cmd, shell=True)
'''