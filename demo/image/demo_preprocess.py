import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.image as mimage
from spipy.image import preprocess
import copy
import sys
import os

if __name__ == "__main__":
	data = np.load("test_adu.npy")
	artif = np.load("artifacts.npy")
	artif = np.where(artif==1)
	artif = np.vstack((artif[0], artif[1])).T
	mask = np.load("mask.npy")
	data[np.isnan(data)] = 0
	data[np.isinf(data)] = 0
	data[data<0] = 0

	# test preprocess.hit_find
	print("\n(1) test preprocess.hit_finding")
	# simulate background
	background = np.random.poisson(10,data.shape[1:])
	non_hit_1 = np.array([background+np.random.randint(1,10,background.shape)])
	non_hit_2 = np.array([np.random.poisson(background*np.random.random(background.shape)*2)])
	data_bg = np.concatenate([data, non_hit_1, non_hit_2])
	index = np.arange(len(data_bg))
	np.random.shuffle(index)
	data_bg = data_bg[index]
	hits = preprocess.hit_find(dataset=data_bg, background=background, radii_range=[None, None, 10, 100], mask=mask, cut_off=10)
	hits2 = preprocess.hit_find_pearson(dataset=data_bg, background=background, radii_range=[None, None, 10, 100], mask=mask, max_cc=0.3)
	print("Predicted hits index by chi-square: " + str(np.where(hits==1)[0]))
	print("Predicted hits index by pearson cc: " + str(np.where(hits2==1)[0]))

	# test preprocess.fix_artifacts
	print("\n(2) test preprocess.fix_artifact")
	ref = copy.deepcopy(data[6])
	data = preprocess.fix_artifact(dataset=data, estimated_center=np.array(data[0].shape)/2, artifacts=artif, mask=mask )
	plt.subplot(1,2,1)
	plt.imshow(np.log(1+np.abs(ref)))
	plt.title('Before fix')
	plt.subplot(1,2,2)
	plt.imshow(np.log(1+np.abs(data[6])))
	plt.title('After fix')
	plt.show()


	print("\n(3) test preprocess.adu2photon")
	adu, newdata = preprocess.adu2photon(dataset=data, mask=mask, photon_percent=0.01, nproc=1, transfer=True, force_poisson=False)
	plt.imshow(np.log(1+newdata[6]))
	plt.title("photon patterns")
	plt.show()


	print("\n(4) test preprocess.fix_artifact_auto")

	if not os.path.isfile("../../../PR_single.h5"):

		pl = mimage.imread('fix_art_auto.png')
		plt.imshow(pl)
		plt.show()
		sys.exit(0)

	else:
		#test_adu.npy")
		data = h5py.File("../../../PR_single.h5", 'r')['adu'][()]
		ref = copy.deepcopy(data)
		newdata = preprocess.fix_artifact_auto(dataset=data, estimated_center=np.array(data[0].shape)/2, njobs=2, mask=mask, vol_of_bins=100)
		for watch in np.random.choice(data.shape[0],10,replace=False):
			plt.subplot(1,2,1)
			plt.imshow(np.log(1+np.abs(ref[watch])))
			plt.title('Before fix')
			plt.subplot(1,2,2)
			plt.imshow(np.log(1+np.abs(newdata[watch])))
			plt.title('After fix')
			plt.show()

	print("\n(5) test preprocess.cal_correction_factor")
	inten_corr = preprocess.cal_correction_factor(det_size=[127,127], polarization='y', detd=581, pixsize=0.6, center=[63,64])
	plt.imshow(inten_corr)
	plt.title("intensity correction factor\n(polarization='y')")
	plt.show()

	print("\n(6) test preprocess.avg_pooling")
	newdata = preprocess.avg_pooling(data, 3, True)
	plt.subplot(1,2,1)
	plt.imshow(np.log(1+data[0]))
	plt.title("Pattern")
	plt.subplot(1,2,2)
	plt.imshow(np.log(1+newdata[0]))
	plt.title("Resized pattern")
	plt.show()

