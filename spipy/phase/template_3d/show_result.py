import numpy as np
import matplotlib.pyplot as plt
import h5py
import sys

from spipy.image import radp
from spipy.analyse import q as qspace

if __name__=="__main__":
	fi = sys.argv[1]
	try:
		exparam = sys.argv[2].split(',')
	except:
		exparam = None
	f = h5py.File(fi,'r')
	prtf = np.abs(np.fft.fftshift(f['PRTF'][()]))
	size = prtf.shape
	prtf_rav = radp.radial_profile(prtf,[size[0]//2,size[1]//2,size[2]//2])
	sr = np.abs(np.fft.fftshift(f['sample retrieved'][()]))
	dr = np.abs(np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(sr))))
	d = np.abs(np.fft.fftshift(f['data'][()]))
	metric = f['convergence metric'][()]
	mod_error = f['modulus error'][()]

	plt.figure(figsize=(20,10))

	plt.subplot(2,3,1)
	plt.imshow(np.log(1+sr[:,size[1]//2,:]))
	plt.title('retrieved (real space)')

	plt.subplot(2,3,2)
	plt.imshow(np.log(1+dr[:,size[1]//2,:]))
	plt.title('retrieved (reciprocal space)')

	plt.subplot(2,3,3)
	plt.imshow(np.log(1+d[:,size[1]//2,:]))
	plt.title('input (reciprocal space)')

	ax1 = plt.subplot(2,3,4)
	plt.plot(metric.mean(axis=0),'-k')
	ax1.set_yscale('log')
	plt.xlabel('steps')
	plt.title('convergence')

	ax2 = plt.subplot(2,3,5)
	plt.plot(mod_error.mean(axis=0),'-r')
	ax2.set_yscale('log')
	plt.xlabel('steps')
	plt.title('modulus error')
	
	ax3 = plt.subplot(2,3,6)
	# q = np.load('resolution.npy')
	qlen = int(np.floor(len(prtf_rav)/np.sqrt(3)))
	if exparam is not None:
		qinfo = qspace.cal_q(float(exparam[0]),float(exparam[1]),float(exparam[2]),float(exparam[3]))
	else:
		qinfo = np.arange(qlen)
	plt.plot(qinfo[:qlen],prtf_rav[:qlen,1],'-k')
	plt.xlabel('q')
	plt.plot(qinfo[:qlen],np.zeros(qlen)+1/np.e,'r--')
	plt.title('PRTF radial average')

	plt.show()
