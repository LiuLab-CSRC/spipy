import sys
import os
import shutil
import argparse
import numpy as np
import yaml
import h5py
from spipy.simulate import sim_adu
from mpi4py import MPI

comm = MPI.COMM_WORLD
m_rank = comm.Get_rank()
m_size = comm.Get_size()

if __name__ == '__main__':

	# parse cmd arguments
	parser = argparse.ArgumentParser(
		allow_abbrev=False,
		description = "Single-particle diffraction simulation.")
	parser.add_argument("-i", "--inpf", type=str, help="Input PDB file.", required=True)
	parser.add_argument("-o", "--outf", type=str, help="Output HDF5 file.", required=True)
	parser.add_argument("-m", "--method", type=str, choices=['atomic','fft'], help="Simulation method, 'atomic' or 'fft'.", required=True)
	parser.add_argument("-n", "--numdata", type=int, help="Number of patterns.", required=True)
	parser.add_argument("-t", "--topol", type=str, help="Detector topology file.", required=True)
	parser.add_argument("--stoprad", type=int, default=0, help="Radius of beam stop, pixel, default=0.")
	parser.add_argument("--fluence", type=float, default=1.0e11, help="Laser fluence (photons/pulse), default=1e15.")
	parser.add_argument("--polarization", type=str, choices=['x','y','none'], default='none', help="Laser polarization direction, 'x' or 'y' or 'none', default='none'.")
	parser.add_argument("--absorption", action="store_true", default=False, help="Set to consider absorption effect into account.")
	parser.add_argument("--scatter_factor", action="store_true", default=False, help="Set to use scattering factor in the simulation.")
	parser.add_argument("--photons", action="store_true", default=False, help="Set to output photon patterns (poisson noise added).")
	parser.add_argument("--projection", action="store_true", default=False, help="Set to output sample projections.")
	parser.add_argument("--verbose", action="store_true", default=False, help="Set to print detailed running log.")
	parser.add_argument("-j", type=int, default=1, help="Number of processes, default=1.")
	args = parser.parse_args()

	# get parameters
	with open(args.topol,'r') as fp:
		topol = yaml.load(fp, Loader=yaml.FullLoader)
	pdb = args.inpf
	method = args.method
	config_param = {'detd' : topol['detd'], 'lambda' : topol['wavelength'], \
					'detsize' : topol['detsize'], 'pixsize' : topol['pixsize'], \
					'stoprad' : args.stoprad, 'polarization' : args.polarization, \
					'num_data' : args.numdata, 'fluence' : args.fluence, \
					'adu_per_eV' : topol['adu_per_eV'], 'detcenter' : topol['center'], \
					'absorption' : args.absorption, 'phy.scatter_factor' : args.scatter_factor, \
					'photons' : args.photons, 'phy.projection' : args.projection, \
					'phy.ram_first' : True}
	euler_range = np.array([[0, np.pi*2], [0, np.pi*2], [0, np.pi*2]])
	save_dir = os.path.abspath(os.path.dirname(args.outf))
	
	# start
	saved_path = sim_adu.go_magic(method=method, save_dir=save_dir, pdb_file=pdb, \
					param=config_param, euler_mode='random', euler_order='zxz', \
					euler_range=euler_range, predefined=None, verbose=args.verbose)

	# move
	if m_size == 1 or m_rank == 0:
		try:
			shutil.move(saved_path, args.outf)
		except Exception as err:
			print(err)
			print("Simulation output is saved as %s" % saved_path)

	# write argv log
	if m_rank == 0:
		with h5py.File(args.outf, "a") as fp:
			fp.create_dataset("information", data=h5py.Empty(int))
			fp["information"].attrs["cmd_line"] = " ".join(sys.argv)











