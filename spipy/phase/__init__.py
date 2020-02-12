__all__ = ['phase3d', 'phase2d', 'phexec', 'phmodel']

from . import phase3d
from . import phase2d
from . import phexec
from . import phmodel

def help(module = None):
	import os
	import sys
	from inspect import isfunction
	flag = 0
	if sys._getframe().f_back.f_code.co_name == 'help':
		flag = 1
	if flag == 0:
		print("this package includes modules-functions : ")
	for f in __all__:
		if flag == 0:
			print("    |- " + f)
			cmd = 'dir(' + f + ')'
			_all_func = eval(cmd)
			for func in _all_func:
				if func[0] != '_' and isfunction(eval(f+'.'+func)):
					print("        |- " + func)
		else:
			print("        |- " + f)
			cmd = 'dir(' + f + ')'
			_all_func = eval(cmd)
			for func in _all_func:
				if func[0] != '_' and isfunction(eval(f+'.'+func)):
					print("            |- " + func)
	"""
	if module is None:
		flag = 0
		if sys._getframe().f_back.f_code.co_name == 'help':
			flag = 1
		if flag == 0:
			print("phase package includes modules : ")
		for f in __all__:
			if flag == 0:
				print("    |- " + f)
			else:
				print("        |- " + f)
	else:
		if module not in __all__:
			print("\n I can't find module " + module + ". Exit")
			return
		print("\n#Function details in module " + module + " :")
		cmd = 'dir(' + module + ')'
		_all_func = eval(cmd)
		ind = 1
		for func in _all_func:
			if func[0] == '_' or not isfunction(eval(module+'.'+func)) :
				continue
			else:
				cmd2 = module + '.' + func + "('help')"
				print("    ")
				print(str(ind) + ") " + func + "(*Input) :")
				eval(cmd2)
				ind += 1
	"""
