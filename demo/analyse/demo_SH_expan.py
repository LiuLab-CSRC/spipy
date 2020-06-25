import numpy as np
import h5py
import matplotlib.pyplot as plt
from spipy.analyse import SH_expan as she

if __name__ == "__main__":
	volume = np.fromfile("volume.bin")
	size = int(np.round(len(volume)**(1.0/3.0)))
	volume.shape = (size, size, size)
	data = {'volume':volume, 'mask':None}
	L = 10
	r = 30
	
	print("\ntest spherical harmonics expansion ...")
	shdes = she.sp_hamonics(data, r=r, L=L)
	plt.plot(range(L+1), shdes, 'o-r')
	plt.xlabel('L')
	plt.ylabel('spherical harmonics expansion')
	plt.show()
