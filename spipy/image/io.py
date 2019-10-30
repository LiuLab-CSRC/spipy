import mrcfile
import numpy as np
import os

def help(module):
	if module=="writeccp4":
		print("Use this function to write volume data into ccp4/mrc format")
		print("    -> Input: volume ( volume data, shape=(Nx,Ny,Nz) )")
		print("              save_file ( output file, str, '...(path)/filename.ccp4' )")
		print("    -> Output: None")
	elif module=="readccp4":
		print("Read ccp4/mrc files")
		print("    -> Input: file_path ( path of file )")
		print("    -> Output: data ( a dict, {'volume':numpy.3darray, 'header':object} )")
	elif module=="pdb2density":
		print("Calculate voxel density map model according to a given pdb file")
		print("    -> Input: pdb_file ( str, the path of pdb file )")
		print("              resolution ( float, the resolution of density map, in Angstrom )")
		print("      option: lamda (None or float, wavelength of laser, in Anstrom, default=None)")
		print("    -> Output: densitymap ( numpy.3darray, voxel model of electron density map )")
	elif module=="cxi_parser":
		print("Print cxi inner path structures")
		print("    -> Input: cxifile ( str, cxi file path )")
		print("      option: out ( str, give 'std' for terminal print or give a file path to redirect to that file)")
		print("    -> Output: None")
	elif module=="xyz2pdb":
		print("Write 3D xyz-coordinates to a pdb file")
		print("    -> Input: xyz_array ( numpy.2darray, shape=(Np,3), colums from the 1st to 3rd is x,y,z coordinates )")
		print("              atom_type ( list, which atoms would like to write in the file. If there is only one item, then all atoms are the same; otherwise you should give a list containing the same number of atom types with the xyz_length. For example, you can either give ['C'] or ['C','H','H','O','H'] for a 5-atom pdb file. No matter upper or lower case)")
		print("      option: b_factor  ( list or None, b_factors to write into pdb, default is None and all b_factors are 1)")
		print("      option: save_file ( str, the complete path of the file that you want to save these information to, default is './convert.pdb' )")
		print("    -> Output: None")
	elif module=="readpdb_full":
		print("Read pdb file and output full infomation")
		print("    -> Input: pdb_file ( str, pdb file path )")
		print("    -> Output: a dict, {res-index:[res-name,{atom-name:[atom-index,atom-mass,x,y,z,occupancy,b-factor]}]}")
	else:
		raise ValueError("No module names "+str(module))


def writeccp4(volume, save_file):
	save_path = os.path.dirname(save_file)
	if save_path != '' and not os.path.exists(save_path):
		raise ValueError("Directory '" + save_path + "' is invalid !")
	if not len(volume.shape)==3:
		raise ValueError("Input volume should be 3D data !")

	vold = np.nan_to_num(volume)
	vold = np.float32(vold)
	with mrcfile.new(save_file, overwrite=True) as mrcf:
		mrcf.set_data(vold)
		mrcf.update_header_from_data()

def readccp4(file_path):
	if not os.path.exists(file_path):
		raise ValueError("Your file path is invalid !")

	data = {'volume':None, 'header':None}
	with mrcfile.open(file_path) as mrcf:
		data['volume'] = mrcf.data
		data['header'] = mrcf.header
	return data

def _readpdb(pdb_file, lamda=None):
	# return array([[index,x,y,z,obs_electron,mass],...]) , shape=(N,6)
	from ..simulate.code import process_pdb
	from collections import OrderedDict
	atom_table_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),'simulate/aux/henke_table')

	atom_types = process_pdb.find_atom_types(pdb_file)
	if lamda is not None:
		eV = process_pdb.wavelength_in_A_to_eV(lamda)
	else:
		eV = None
	# scatt_info =  {atom:[number, f0, mass], ...}
	scatt_info = process_pdb.make_scatt_list(atom_types, atom_table_path, eV)
	atoms = process_pdb.get_atom_coords(pdb_file, scatt_info)
	# process symmetry
	(s_l, t_l)  = process_pdb.read_symmetry(pdb_file)
	if len(s_l)*len(t_l)>0:
		all_atoms = process_pdb.apply_symmetry(atoms, s_l, t_l)
	else:
		all_atoms = atoms
	return all_atoms

def readpdb_full(pdb_file):
	# return {res-index:[res-name,{atom-name:[atom-index,atom-mass,x,y,z,occupancy,b-factor], ...}], ...}
	from ..simulate.code import process_pdb
	atom_table_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),'simulate/aux/henke_table')

	with open(pdb_file, 'r') as fp:
		lines = fp.readlines()
	pdb = {}
	for line in lines:
		if not line.startswith("ATOM"):
			continue
		a_name = line[11:17].strip()
		res_name = line[17:20].strip()
		res_index = int(line[22:26])
		x = float(line[30:38])
		y = float(line[38:46])
		z = float(line[46:54])
		occup = float(line[54:60])
		b_factor = float(line[60:66])
		atom = line[66:].strip()
		if len(atom)>2 or len(atom)==0:
			atom = atom[:len(atom)-2]
		if len(atom) > 0:
			a_index, a_mass = process_pdb.find_mass(atom_table_path, atom)
		else:
			a_index = -1
			a_mass = -1
		if res_index not in pdb.keys():
			pdb[res_index] = [res_name, {}]
		pdb[res_index][1][a_name]=[a_index, a_mass, x, y, z, occup, b_factor]
	return pdb


def pdb2density(pdb_file, resolution, lamda=None):
	from ..simulate.code import process_pdb
	all_atoms = _readpdb(pdb_file, lamda)    # shape=(N,6)
	# process density
	density = process_pdb.atoms_to_density_map(all_atoms, resolution)
	density = np.abs(process_pdb.low_pass_filter_density_map(density))
	# put it at center
	points = np.array(np.where(density>1))
	center = np.round(np.mean(points,axis=1)).astype(int)
	R = int(np.ceil(np.linalg.norm(points.T-center, axis=1).max()))
	ext_part = density[max(center[0]-R,0):min(center[0]+R,density.shape[0]),\
						max(center[1]-R,0):min(center[1]+R,density.shape[1]),\
						max(center[2]-R,0):min(center[2]+R,density.shape[2])]
	ext_cen = np.array(ext_part.shape)//2
	box = np.zeros((2*R+1,2*R+1,2*R+1))
	box[R-ext_cen[0]:R+ext_part.shape[0]-ext_cen[0],\
		R-ext_cen[1]:R+ext_part.shape[1]-ext_cen[1],\
		R-ext_cen[2]:R+ext_part.shape[2]-ext_cen[2]] = ext_part

	return box

def xyz2pdb(xyz_array, atom_type, b_factor=None, save_file="./convert.pdb"):
	if xyz_array.shape[1]!=3 or len(xyz_array.shape)!=2 or type(atom_type)!=list:
		raise ValueError('Invalid input data shape/type !')
	if b_factor is not None and len(b_factor)!=len(xyz_array):
		raise ValueError('The length of b_factor should be equal to xyz_array')
	xyz = xyz_array
	pdb = open(save_file,'w')
	for ind,line in enumerate(xyz):
		atom = "%5d" %(ind+1)
		resi = "%4d" %1
		x = "%+8.3f" %line[0]
		y = "%+8.3f" %line[1]
		z = "%+8.3f" %line[2]
		occup = "%6.2f" %1.0
		if b_factor is not None:
			t = "%6.2f" %b_factor[ind]
		else:
			t = "%6.2f" %1.0
		if len(atom_type)==1:
			atype = "%-2s" %(atom_type[0].upper())
		elif len(atom_type)==xyz_array.shape[0]:
			atype = "%-2s" %(atom_type[ind].upper())
		else:
			raise ValueError("length of atom_type does not match length of xyz_array")
		content = 'ATOM'+'  '+atom+'  '+atype+'  '+'ALA'+'  '+resi+\
					'    '+x+y+z+occup+t+'          '+atype+'\n'
		pdb.writelines(content)
	pdb.close()

class _CXIDB():

	def print_path(self, d, groups, depth):
		for gname in groups:
			g = d[gname]
			if str(type(g)).split('.')[-2]=="group":
				print(" "*depth+"|--"+gname)
				children = g.keys()
				self.print_path(g, children, depth+3)
			elif str(type(g)).split('.')[-2]=="dataset":
				print(" "*depth+"|--"+gname)
			else:
				continue

	def parser(self, cxifile, stdout='std'):
		import sys
		import h5py
		f = h5py.File(cxifile,'r')
		groups = f.keys()
		depth = 3
		if stdout!='std':
			nf = open(stdout,'w')
			oldstd = sys.stdout
			sys.stdout = nf
		print(cxifile)
		self.print_path(f, groups, depth)
		if stdout!='std':
			sys.stdout = oldstd
			nf.close()
		f.close()

def cxi_parser(cxifile, out='std'):
	cxidb = _CXIDB()
	cxidb.parser(cxifile, out)