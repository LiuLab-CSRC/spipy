import numpy as np
import sys
import os
import gc
from ..analyse import saxs
from . import radp

def help(module):
	if module=="cluster_fSpec":
		print("This function is used to do single-nonsingle hits clustering using linear/non-linear decomposition and spectural clustering")
		print("    -> Input: dataset (numpy.ndarray, shape=(Nd,Nx,Ny)")
		print("      option: low_filter (float 0~1, the percent of area at the frequency center that is used\
							 for clustering, default=0.3)")
		print("      option: decomposition (str, decoposition method, choosen from 'LLE', 'SVD' and 'SpecEM'\
											default='SVD')")
		print("      option: ncomponent (int, number of components left after decomposition, default=2)")
		print("      option (LLE): nneighbors (int, number of neighbors in LLE graph, default=10)")
		print("      option (LLE): LLEmethod (methods used in LLE, choosen from 'standard', 'modified', 'hessian' and 'ltsa',\
									 default='standard')")
		print("      option: clustering (int, whether to do clustering (<0 or >0) and how many classes (value of this param) to have)")
		print("      option: njobs (int, number of jobs)")
		print("      option: verbose (bool, whether to print details, default=True)")
		print("    -> Return: list, [data_after_decomposition, predicted_labels]")
		print("[Notice] The input dataset is not recommended to contain more than 1k patterns, but it's also neccessary to have more than 50 ones.\
You can split the original dataset into several parts and use multi-processors to deal with them.")
		print("Help End. Exit.")
		return
	elif module=="cluster_fTSNE":
		print("This function is used to do single-nonsingle patterns clustering using TSNE and kmeans")
		print("    -> Input: dataset (numpy.ndarray, shape=(Nd,Nx,Ny)")
		print("      option: low_filter (float 0~1, the percent of area at the frequency center that is used for clustering, default=0.3)")
		print("      option (TSNE): no_dims (+int, dimensions after decomposition, default=2)")
		print("      option (TSNE): perplexity (+int, perlexity value to evaluate P(i|j) in TSNE, default=20)")
		print("      option (TSNE): use_pca (bool, whether to use PCA to generate initiate features, default=True)")
		print("      option (TSNE): initial_dims (+int, output dimensions of inititate PCA, ignored if use_pca=False, default=50)")
		print("      option (TSNE): max_iter (+int, max iterations, default=1000, suggested >500)")
		print("      option (TSNE): theta (0~1 float, the speed vs accuracy trade-off parameter, theta=1 means highest speed, default=0.5)")
		print("      option (TSNE): randseed (int, >=0 use 'randseed' as initiate value's generating seed, <0 use current time as random seed, default=-1)")
		print("      option: clustering (int, whether to do clustering (<0 or >0) and how many classes (value of this param) to have)")
		print("      option: njobs (number of threads in parallel, default=1)")
		print("      option: verbose (bool, whether to print details, default=True)")
		print("    -> Return: list, [data_after_decomposition, predicted_labels]")
		print("[Notice] The input dataset is not recommended to contain more than 1k patterns, but it's also neccessary to have more than 50 ones.\
You can split the original dataset into several parts and use multi-processors to deal with them.")
		print("Help End. Exit.")
		return
	elif module=="diffusion_map":
		print("This function is used to do Diffusion Map embedding and get features")
		print("    -> Input: dataset (numpy.ndarray, shape=(Nd,Nx,Ny) or (Nd,Npix)")
		print("              nEigs (positive int, number of dimensions to be left on eigenvectors)")
		print("      option: neigh_kept (int, number of nearest neighbors to be kept in D [distance] matrix, default=100)")
		print("      option: sigma_opt (float, kernel matrix K=exp(-D/mean(D)/sigma_opt), try 0.1~100, default=1)")
		print("      option: alpha (float, adjustment factor on kernel matrix, default=1)")
		print("    -> Return: list, [eigenvalue, eigenvector], eigenvalue is a (nEigs,) ndarray, eigenvector is a (Nd,nEigs) ndarray")
	else:
		raise ValueError("No module names "+str(module))

def cluster_fSpec(dataset, low_filter=0.3, decomposition='SVD', ncomponent=2, nneighbors=10, LLEmethod='standard', clustering=2, njobs=1, verbose=True):
	if decomposition not in ['LLE', 'SVD', 'SpecEM']:
		raise RuntimeError("I can't recognize the decomposition method.")
	if decomposition=="LLE" and LLEmethod not in ['standard', 'modified', 'hessian','ltsa']:
		raise RuntimeError("I can't recognize the LLE method.")
	#import saxs
	#import radp
	ncomponent = abs(int(ncomponent))
	nneighbors = abs(int(nneighbors))
	clustering = abs(int(clustering))
	njobs = abs(int(njobs))
	rcenter = [int(dataset.shape[1]*low_filter/2.0), int(dataset.shape[2]*low_filter/2.0)]
	# fft
	if verbose:
		print("\nStart FFT analysis ...")
	dataset[np.where(dataset<0)] = 0
	dataset[np.isnan(dataset)] = 0
	dataset[np.isinf(dataset)] = 0
	fdataset = np.zeros(dataset.shape)
	for ind,data in enumerate(dataset):
		fdataset[ind] = np.abs(np.fft.fftshift(np.fft.fft2(data)))
		if verbose:
			sys.stdout.write("Processing " + str(ind) + "/" + str(len(dataset)) + " ...\r")
			sys.stdout.flush()
	if verbose:
		print("\nDone.")
		print("\nStart normalization ...")
	# normalization
	center_data = (fdataset.shape[1]//2, fdataset.shape[2]//2)
	fdataset = fdataset[:, center_data[0]-rcenter[0]:center_data[0]+rcenter[0], center_data[1]-rcenter[1]:center_data[1]+rcenter[1]]
	center_data = (fdataset.shape[1]//2, fdataset.shape[2]//2)
	saxs_data = saxs.cal_saxs(fdataset)
	saxs_intens = radp.radial_profile(saxs_data, center_data, None)
	dataset_norm = np.zeros(fdataset.shape)
	for ind,pat in enumerate(fdataset):
		pat_normed = radp.radp_norm(saxs_intens[:,1], pat, center_data, None)
		dataset_norm[ind] = pat_normed
		if verbose:
			sys.stdout.write("Processing " + str(ind) + "/" + str(len(fdataset)) + " ...\r")
			sys.stdout.flush()
	if verbose:
		print("\nDone.")
		print("\nStart clustering...")
	# Spectral clustering
	from sklearn.cluster import SpectralClustering
	from sklearn.decomposition import TruncatedSVD
	from sklearn.manifold import SpectralEmbedding
	from sklearn.manifold import LocallyLinearEmbedding
	dataset_norm.shape = (dataset_norm.shape[0], dataset_norm.shape[1]*dataset_norm.shape[2])

	# decomposition
	log_data_norm = np.log(1+np.abs(dataset_norm))
	if decomposition=='LLE':
		decomp = LocallyLinearEmbedding(n_neighbors=nneighbors, method=LLEmethod, n_components=ncomponent, n_jobs=njobs)
	elif decomposition=='SVD':
		decomp = TruncatedSVD(n_components=ncomponent)
	elif decomposition=='SpecEM':
		decomp = SpectralEmbedding(n_components=ncomponent, eigen_solver='arpack', n_jobs=njobs)
	dataset_decomp = decomp.fit_transform(log_data_norm)

	# return dataset_decomp
	if clustering>0:
		cluster = SpectralClustering(n_clusters=clustering, affinity='rbf', n_jobs=njobs)
		label = cluster.fit_predict(dataset_decomp)
		return dataset_decomp,label
	else:
		return dataset_decomp,[]


def cluster_fTSNE(dataset, low_filter=0.3, no_dims=2, perplexity=20, use_pca=True, initial_dims=50, max_iter=500, theta=0.5, randseed=-1, clustering=2, njobs=1, verbose=True):
	
	from sklearn.decomposition import PCA
	sys.path.append(os.path.join(os.path.dirname(__file__),'bhtsne_source'))
	from .bhtsne_source import bhtsne

	no_dims = abs(int(no_dims))
	initial_dims = abs(int(initial_dims))
	max_iter = abs(int(max_iter))
	clustering = abs(int(clustering))
	theta = min(np.abs(theta),1)
	rcenter = [int(dataset.shape[1]*low_filter/2.0), int(dataset.shape[2]*low_filter/2.0)]
	# fft
	if verbose:
		print("\nStart FFT analysis ...")
	dataset[np.where(dataset<0)] = 0
	dataset[np.isnan(dataset)] = 0
	dataset[np.isinf(dataset)] = 0
	fdataset = np.zeros(dataset.shape)
	for ind,data in enumerate(dataset):
		fdataset[ind] = np.abs(np.fft.fftshift(np.fft.fft2(data)))
		if verbose:
			sys.stdout.write("Processing " + str(ind) + "/" + str(len(dataset)) + " ...\r")
			sys.stdout.flush()
	if verbose:
		print("\nDone.")
		print("\nStart normalization ...")
	# normalization
	center_data = (fdataset.shape[1]//2, fdataset.shape[2]//2)
	fdataset = fdataset[:, center_data[0]-rcenter[0]:center_data[0]+rcenter[0], center_data[1]-rcenter[1]:center_data[1]+rcenter[1]]
	center_data = (fdataset.shape[1]//2, fdataset.shape[2]//2)
	saxs_data = saxs.cal_saxs(fdataset)
	saxs_intens = radp.radial_profile(saxs_data, center_data, None)
	dataset_norm = np.zeros(fdataset.shape)
	for ind,pat in enumerate(fdataset):
		pat_normed = radp.radp_norm(saxs_intens[:,1], pat, center_data, None)
		dataset_norm[ind] = pat_normed
		if verbose:
			sys.stdout.write("Processing " + str(ind) + "/" + str(len(fdataset)) + " ...\r")
			sys.stdout.flush()
	if verbose:
		print("\nDone.")
		print("\nStart decomposition using TSNE ...")
	# decomposition
	dataset_norm.shape = (dataset_norm.shape[0], dataset_norm.shape[1]*dataset_norm.shape[2])
	log_data_norm = np.log(1+np.abs(dataset_norm))
	del dataset_norm
	del fdataset
	del saxs_data
	gc.collect()
	if use_pca:
		pca = PCA(n_components=initial_dims)
		log_data_norm = pca.fit_transform(log_data_norm)
		del pca
	embedding_array = bhtsne.run_bh_tsne(log_data_norm, no_dims=no_dims, perplexity=perplexity, use_pca=False, initial_dims=initial_dims, max_iter=max_iter, theta=theta, randseed=randseed, verbose=verbose)
	# clustering
	if clustering>0:
		if verbose:
			print("\nStart clustering ...")
		from sklearn import cluster
		centroid, label, inertia = cluster.k_means(embedding_array, clustering, n_jobs=njobs)
		return embedding_array, label
	else:
		return embedding_array, []


def diffusion_map(dataset, nEigs, neigh_kept=100, sigma_opt=1, alpha=1):
	'''
		dataset : (Nd,Nx,Ny) or (Nd,Np)
	'''
	if len(dataset.shape) not in [2,3]:
		raise ValueError("Input dataset should be a 2 or 3-dimensional array !")
	if neigh_kept > len(dataset):
		raise ValueError("The neigh_kept should not be larger than data amount")

	nEigs = abs(int(nEigs))
	if len(dataset.shape) == 3:
		X = np.matrix( dataset.reshape([dataset.shape[0],dataset.shape[1]*dataset.shape[2]]), dtype=np.float64).T # (Np, Nd)
	else:
		X = np.matrix(dataset).T

	# calculate Dij
	Nd = X.shape[1]
	D = np.tile( np.sum(np.multiply(X,X), axis=0), (Nd, 1) )
	D = D - X.T * X
	D = D + D.T
	D = np.abs(D + D.T) / 2

	# distance function
	mean = np.mean(D)
	K = np.exp(-D / (mean * sigma_opt))
	
	# spasify
	nneigh = int(neigh_kept)
	largest_inx_col = np.argsort(-K, axis=1)[:, :nneigh]
	largest_inx_row = np.tile(np.matrix(np.arange(Nd)).T, (1,nneigh))
	tmp = np.matrix(np.zeros(K.shape))
	tmp[largest_inx_row, largest_inx_col] = K[largest_inx_row, largest_inx_col]
	K = np.maximum(tmp, tmp.T)
	
	# kernel
	K = np.divide(K, np.sum(K,1) * np.sum(K,0))
	K = np.power(K, alpha)
	W = np.sum(np.array(K),axis=0)
	W = 1 / np.sqrt(W)
	W = np.matrix(np.diag(W))
	L = W * K * W

	# eig
	eigval, eigvec = np.linalg.eig(L)
	eigval = np.real(eigval)
	eigvec = np.real(eigvec)
	eig_inx = np.argsort(-eigval)[:nEigs+1]
	eigval = eigval[eig_inx][1:]
	eigvec = np.array(eigvec[:,eig_inx]).T
	eigvec /= (eigvec[0]+1e-20) 
	eigvec = eigvec[1:,:]
	
	return eigval, eigvec.T
