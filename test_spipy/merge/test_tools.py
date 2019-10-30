import numpy as np
from spipy.merge import tools
from spipy.analyse import q
from spipy.image import preprocess
import matplotlib.pyplot as plt
import time

if __name__ == '__main__':
	
	fpath = '../phase/3dvolume.bin'
	det_size = [95,95]
	# make mask
	mask = np.zeros(det_size)
	mask[46:49,:] = 1

	q_coor, _, _, _ = q.ewald_mapping(581, 7.9, 0.87, det_size, None)
	q_coor = q_coor.T.reshape((3,-1))

	print("\n-- Generate 2000 quaternions ..")
	quats = tools.get_quaternion(Num_level=20)[np.random.choice(16000,2000,replace=False)]
	
	print("\n-- Generate 2000 slices from 3D model ..")
	model_0 = np.fromfile(fpath).reshape((125,125,125))
	t1 = time.time()
	slices = tools.get_slice(model=model_0, rotations=quats, det_size=det_size, \
									det_center=None, mask=mask, slice_coor_ori=q_coor)
	print("Done. Time : "+str(time.time()-t1)+" s")
	plt.subplot(1,2,1)
	plt.imshow(np.log(1+slices[0]))
	plt.title('pattern index=0')
	plt.subplot(1,2,2)
	plt.imshow(np.log(1+slices[1999]))
	plt.title('pattern index=1999')
	plt.show()

	print("\n-- Merge the generated 2000 patterns into a new model ..")
	model_1 = np.zeros(model_0.shape)
	t1 = time.time()
	tools.merge_slice(model=model_1, rotations=quats, slices=slices, weights=None, \
									det_center=None, mask=mask, slice_coor_ori=q_coor)
	print("Done. Time : "+str(time.time()-t1)+" s")
	plt.figure(figsize=(10,3))
	plt.subplot(1,3,1)
	plt.imshow(np.log(1+model_1[62,:,:]))
	plt.title('Y-Z Plain of Merged Model')
	plt.subplot(1,3,2)
	plt.imshow(np.log(1+model_1[:,62,:]))
	plt.title('X-Z Plain of Merged Model')
	plt.subplot(1,3,3)
	plt.imshow(np.log(1+model_1[:,:,62]))
	plt.title('X-Y Plain of Merged Model')
	plt.show()
	
	print("\n-- Calculate poisson likelihood between slices and pattern ..")
	pat = tools.get_slice(model=model_0, rotations=quats[1000], det_size=det_size, det_center=None, mask=mask)
	adu, pat = preprocess.adu2photon(dataset=np.array([pat]), mask=mask, photon_percent=0.1, nproc=1, transfer=True, force_poisson=True)
	R_jk = np.zeros(len(slices))
	for ind, s in enumerate(slices):
		R_jk[ind] = tools.poisson_likelihood(W_j=s, K_k=pat[0], beta=5, weight=None)
	P_jk = R_jk/np.sum(R_jk)
	plt.hist(P_jk, bins=50)
	plt.title("Normalized poisson-likelihood probabilities (pattern_index=999)")
	plt.show()

	print("\n-- Do maximization ..")
	# use P_jk for ease
	W_j_prime = tools.maximization(slices.reshape(2000,np.product(det_size)), P_jk)
	plt.imshow(np.log(1+W_j_prime.reshape(det_size)))
	plt.show()