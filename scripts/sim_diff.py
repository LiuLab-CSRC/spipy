import sys
import os
import shutil
import argparse
import numpy as np
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
	parser.add_argument("-i", "--inpf", type=str, help="Input PDB file", required=True)
	parser.add_argument("-o", "--outf", type=str, help="Output HDF5 file", required=True)
	parser.add_argument("-m", "--method", type=str, choices=['atomic','fft'], help="simulation method, 'atomic' or 'fft'", required=True)
	parser.add_argument("-n", "--numdata", type=int, help="number of patterns", required=True)
	parser.add_argument("--detd", type=float, help="detector distance, mm", required=True)
	parser.add_argument("--wavelength", type=float, help="laser wave length, angstrom", required=True)
	parser.add_argument("--detsize", type=int, help="detector size, pixel", required=True)
	parser.add_argument("--pixsize", type=float, help="pixel side length, mm", required=True)
	parser.add_argument("--stoprad", type=int, default=0, help="radius of beam stop, pixel, default=0")
	parser.add_argument("--fluence", type=float, default=1.0e15, help="laser fluence (photons/pulse), default=1e15")
	parser.add_argument("--polarization", type=str, choices=['x','y','none'], default='none', help="laser polarization direction, 'x' or 'y' or 'none', default='x'")
	parser.add_argument("--absorption", action="store_true", default=False, help="set to consider absorption effect into account")
	parser.add_argument("--scatter_factor", action="store_true", default=False, help="set to use scattering factor in the simulation")
	parser.add_argument("--photons", action="store_true", default=False, help="set to output photon patterns (poisson noise added)")
	parser.add_argument("--projection", action="store_true", default=False, help="set to output sample projections")
	parser.add_argument("--verbose", action="store_true", default=False, help="set to print detailed running log")
	parser.add_argument("-j", type=int, default=1, help="number of processes, default=1")
	args = parser.parse_args()

	# get parameters
	pdb = args.inpf
	method = args.method
	config_param = {'detd' : args.detd, 'lambda' : args.wavelength, \
					'detsize' : args.detsize, 'pixsize' : args.pixsize, \
					'stoprad' : args.stoprad, 'polarization' : args.polarization, \
					'num_data' : args.numdata, 'fluence' : args.fluence, \
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














