from spipy.analyse import criterion
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
	'''
		test criterion
	'''

	f1 = 'volume.bin'
	f2 = 'volume_rotated.bin'
	d1 = np.fromfile(f1).reshape((125,125,125))
	d2 = np.fromfile(f2).reshape((125,125,125))
	rlist = np.arange(60)

	# r-factor
	rfac = criterion.r_factor(d1, d2)
	rfac_l = criterion.r_factor_shell(d1, d2, rlist)
	print("r-factor   : %f" % rfac)

	# fsc
	fsc = criterion.fsc(d1, d2, rlist)

	# r-split
	rsplit = criterion.r_split(d1, d2, rlist)

	# cc
	cc = criterion.Pearson_cc(d1, d2, 0)
	print("pearson-cc : %f" % cc)

	# prtf
	ph_ang_1 = np.random.normal(np.ones(d1.shape), 0.1)
	ph_ang_2 = np.random.normal(np.ones(d1.shape), 0.1)
	d1_1 = d1 * np.exp(1j*ph_ang_1)
	d1_2 = d1 * np.exp(1j*ph_ang_2)
	prtf = criterion.PRTF(np.array([d1_1,d1_2]), [62,62,62], None)
	
	plt.subplot(2,2,1)
	plt.plot(rlist, rfac_l, 'r-')
	plt.title("R-factor")
	
	plt.subplot(2,2,2)
	plt.plot(rlist, fsc, 'b-')
	plt.title("FSC")

	plt.subplot(2,2,3)
	plt.plot(rlist, rsplit, 'k-')
	plt.title("R-split")

	plt.subplot(2,2,4)
	plt.plot(prtf[:60,0], prtf[:60,1], 'm-')
	plt.title("PRTF")

	plt.show()