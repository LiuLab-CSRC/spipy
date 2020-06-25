import numpy as np
import h5py
import sys
from spipy.image import io
from spipy.image import radp
from spipy.image import quat
import matplotlib.pyplot as plt

if __name__ == '__main__':
	
	# test radp.radial_profile and radp.radp_norm
	print("\nTest radp ...")
	d = np.fromfile("../analyse/volume.bin").reshape((125,125,125))
	d_noise = np.log(1+d)
	center = np.array(d.shape)//2
	Iq = radp.radial_profile(d, center, None)[:,1]
	d_denoise = radp.radp_norm(Iq, d_noise, center, None)
	err_0 = np.sqrt(np.sum((d-d_noise)**2))
	err_1 = np.sqrt(np.sum((d-d_denoise)**2))
	print("Error before normalization : %f , and after : %f" % (err_0, err_1))

	# test io.readpdb_full, io.pdb2density, io.xyz2pdb
	print("\nTest io.readpdb_full ...")
	pdbfile = "../simulate/1N0U1.pdb"
	pdb = io.readpdb_full(pdbfile)
	for items in pdb.items():
		print(items)
		break
	print("...")

	print("\nTest io.pdb2density ...")
	den = io.pdb2density(pdbfile, 2.0, 1.0)
	c = len(den)//2
	plt.imshow(den[c,:,:])
	plt.show()

	print("\nTest io.xyz2pdb ...")
	xyz = np.random.rand(10,3) * 10
	typ = ['CA', 'CB', 'CB', 'H', 'O', 'H', 'S', 'N', 'N','C']
	io.xyz2pdb(xyz, typ, None ,"convert.pdb")
	print("Check convert.pdb")

	print("\nTest io.writeccp4 ...")
	io.writeccp4(den, "1N0U1.ccp4")
	print("Check 1N0U1.ccp4")

	print("\nTest io.readccp4 ...")
	data = io.readccp4("1N0U1.ccp4")
	print("Header : %s" % (data['header']))
	plt.imshow(data['volume'][c,:,:])

	# test cxiparser
	print("\nTest io.cxi_parser ..")
	io.cxi_parser("../test_pattern.h5")

	# test quat
	q0 = np.array([0.688191,-0.262866,-0.525731,-0.425325])
	v0 = np.array([1, -1, 1])
	print("\nTest quat ...")
	print("q0 = %s" % str(q0))
	q1 = quat.invq(q0)
	print("q1 = invq(q0) = %s" % str(q1))
	q2 = quat.quat_mul(q0, q1)
	print("q2 = q0 * q1 = %s" % str(q2))
	q3 = quat.conj(q0)
	print("q3 = conj(q0) = %s" % str(q3))
	az = quat.quat2azi(q0)
	print("az = quat2azi(q0) = %s" % str(az))
	q4 = quat.azi2quat(az)
	print("q4 = azi2quat(az) = %s" % str(q4))
	rt = quat.quat2rot(q0)
	print("rt = quat2rot(q0) = \n%s" % str(rt))
	q5 = quat.rot2quat(rt)
	print("q5 = rot2quat(rt) = %s" % str(q5))
	v1 = quat.rotv(v0, q0)
	print("v0 = %s" % str(v0))
	print("v1 = rotv(v0, q0) = %s" % str(v1))
	q6 = quat.Slerp(q0, q1, 0.5)
	print("q6 = Slerp(q0, q1, 0.5) = %s" % str(q6))










