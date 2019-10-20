import numpy as np
import sys
from ..image import radp
from . import q as spi_q

def help(module):
	if module=="grid":
		print("This finction is used to calculate grid of a pattern")
		print("    -> Input: input_patt (numpy.ndarray, shape = (Nx,Ny))")
		return
	elif module=="friedel_search":
		print("This function is used to find the zeros frequency point of an SPI pattern")
		print("    -> Input: pattern (none negative numpy.ndarray, shape=(Nx, Ny))")
		print("              estimated_center (estimated center of the pattern : (Cx, Cy), error in 20 pixels )")
		print("      option: mask (0/1 binary pattern, shape=(Nx, Ny), 1 means masked area, 0 means useful area, default=None)")
		print("      option: small_r (int, radius of search area for center allocation candidates, pixel)")
		print("      option: large_r (int, radius of area for sampling frediel twin points, pixel)")
		print("    -> Output: center, a numpy 1d array, [cx, cy], int")
		return
	elif module=="inten_profile_vaccurate":
		print("This finction is used to calculate accumulate intensity profile of SPI patterns")
		print("    -> Input: dataset (numpy.ndarray, shape=(Nd,Nx,Ny)) ")
		print("              mask (0/1 binary pattern, shape=(Nx, Ny), 1 means masked area, 0 means useful area)")
		print("              *exp_param (detd (mm) , lamda (A), det_r (pixel), pixsize (mm))")
		print("[Notice] We don't recommend you to use this function as it is very slow. Use 'inten_profile_vfast' instead.")
		return
	elif module=="inten_profile_vfast":
		print("This finction is used to calculate accumulate intensity profile of SPI patterns")
		print("The patterns stored in .h5 file should be a ndarray (num, Nx, Ny)")
		print("    -> Input: dataset (numpy.ndarray, shape=(Nd,Nx,Ny)) ")
		print("              mask (0/1 binary pattern, shape=(Nx, Ny), 1 means masked area, 0 means useful area)")
		print("              *exp_param (detd (mm) , lamda (A), det_r (pixel), pixsize (mm))")
		return
	elif module=="cal_saxs":
		print("This finction is used to calculate the saxs pattern of an SPI data set")
		print("    -> Input: data (patterns, numpy.ndarray, shape=(Nd,Nx,Ny)) ")
		return
	elif module=="centering":
		print("This function shift the input pattern to its center. Output pattern is a part of original one.")
		print("    -> Input: pat (input pattern, a 2d numpy array, shape=(Nx,Ny))")
		print("              estimated_center (estimated center of original pattern, )")
		print("      option: mask (0/1 binary pattern, shape=(Nx, Ny), 1 means masked area, 0 means useful area, default=None)")
		print("      option: large_r / small_r (see 'spipy.analyse.saxs.friedel_search' help)")
		print("    -> Output: newpat ( output pattern, a 2d numpy array, shape=(Nx',Ny'))")
		print("               newmask ( ONLY if mask!=None, then the cut mask pattern will be output)")
	elif module=="particle_size":
		print("This function uses self-correlation transformation method to estimate particle size of saxs pattern")
		print("    -> Input: saxs (saxs pattern, numpy.2darray, shape=(Nx,Ny))")
		print("              estimated_center (the estimated center of saxs pattern)")
		print("     #option: exparam (experimetnal parameters, str, 'detector-sample distance(mm),lambda(A),pixel length(mm)', such as '578,7.9,0.3', default=None)")
		print("     #option: high_filter_cut (float between (0,1), which determine the FWHM of high pass filter, larger value means smaller filter width, default=0.3)")
		print("     #option: power (positive float 0~1, a power conducted on pattern to enhance the contribution of high q data, default=0.7)")
		print("     #option: mask (0/1 binary pattern, shape=(Nx, Ny), 1 means masked area, 0 means useful area, default=None)")
		print("    -> Output: list, [particle_size(float), auto_correlation_radial_profile(2darray)]. particle size is not garunteed to be correct, \
			RECOMMOND to watch auto-correlation radial profile and find the locations of peaks, which are most trustable values of possible particle sizes. \
			auto_correlation_radial_profile is a 2-d array, where the 1st colum is particle size in nm (if exparam is given) and 2nd colum is the auto-correlation value.")
		print("[NOTICE] is your give exparam, the output particle_size is in nanometer; otherwise it is in pixels. Auto correlation radial profile is in pixels")
	elif module=="particle_size_sp":
		print("This function fit the diameter of spherical-like samples by using q0 values (the first intensity minimum) of their SPI patterns")
		print("    -> Input: dataset (patterns, numpy array, shape=(Nd, x, y))")
		print("              exparam (set up parameters of experiment, a list [detector-distance(mm), lambda(A), pixel-length(mm)])")
		print("              fitarea (a list [nr, nR], define an ROI where the radius is between nr and nR, and use it to do the fitting)")
		print("              badsearchr (int, the largest distance from the 10th Iq-peaks to center within all patterns, unit=pixel)")
		print("              method (fitting method, str, chosen from 'q0' and 'lsq', default = 'q0')")
		print("     #option: mask (0/1 binary pattern, shape=(Nx, Ny), 1 means masked area, 0 means useful area, default=None)")
		print("     #option: center (center of diffration patterns, list/array, default = None and the program will search center automatically for every pattern)")
		print("     #option: verbose (bool, whether to display progress bar, default = True)")
	else:
		raise ValueError("No module names "+str(module))


# generate grid from input pattern's shape
def grid(input_patt):	
	size = input_patt.shape
	if len(size)==2:
		return np.mgrid[0:size[0],0:size[1]]
	elif len(size)==3:
		return np.mgrid[0:size[0],0:size[1],0:size[2]]
	else:
		return

# frediel search of SPI pattern to find center
def friedel_search(pattern, estimated_center, mask=None, small_r=None, large_r=None):
	if small_r is not None and int(small_r)<=0:
		raise ValueError("small_r should be a positive integer or None")
	if large_r is not None and int(large_r)<=0:
		raise ValueError("large_r should be a positive integer or None")

	if mask is not None:
		maskpattern = pattern * (1 - mask)
	else:
		maskpattern = pattern
		
	size = np.array(maskpattern.shape)
	estimated_center = np.array(estimated_center).astype(int)
	if small_r is None:
		search_z = [max(size[0]//20, 20), max(size[1]//20, 20)]
	else:
		search_z = [int(small_r)*2, int(small_r)*2]
	searchzone_top = max(0, estimated_center[0]-search_z[0]//2)
	searchzone_bottom = min(size[0], estimated_center[0]+search_z[0]//2)
	searchzone_left = max(0, estimated_center[1]-search_z[1]//2)
	searchzone_right = min(size[1], estimated_center[1]+search_z[1]//2)
	search_zone = np.mgrid[searchzone_top : searchzone_bottom,\
					searchzone_left : searchzone_right]
	if large_r is None:
		fred_z = [min(max(size[0]//2, 50),size[0]), min(max(size[1]//2, 50),size[1])]
	else:
		fred_z = [int(large_r)*2, int(large_r)*2]
	score = np.inf
	center = np.array([0,0], dtype=int)

	for cen in search_zone.T.reshape(np.product(search_zone[0].shape),2):
		fred_zone = np.mgrid[cen[0]-fred_z[0]//2:cen[0]+fred_z[0]//2,cen[1]-fred_z[1]//2:cen[1]+fred_z[1]//2]
		fred_zone = fred_zone.reshape((2,fred_zone.shape[1]*fred_zone.shape[2])).astype(int)
		inv_fred_zone = np.array([2*cen[0] - fred_zone[0], 2*cen[1] - fred_zone[1]], dtype=int)
		if mask is not None:
			no_mask_area = (mask[fred_zone[0],fred_zone[1]]==0) & (mask[inv_fred_zone[0],inv_fred_zone[1]]==0)
			this_score = 2 * np.sum(np.abs(maskpattern[fred_zone[0],fred_zone[1]] - maskpattern[inv_fred_zone[0], inv_fred_zone[1]])\
					* no_mask_area.astype(float)) \
					/ np.sum(np.abs(maskpattern[fred_zone[0],fred_zone[1]] + maskpattern[inv_fred_zone[0], inv_fred_zone[1]])\
					 * no_mask_area.astype(float))
			this_score /= len(np.where(no_mask_area)[0])
		else:
			this_score = 2 * np.sum(np.abs(maskpattern[fred_zone[0],fred_zone[1]] - maskpattern[inv_fred_zone[0], inv_fred_zone[1]])) \
					/ np.sum(np.abs(maskpattern[fred_zone[0],fred_zone[1]] + maskpattern[inv_fred_zone[0], inv_fred_zone[1]]))
			this_score /= len(fred_zone[0])
		if score>this_score:
			center = cen
			score = this_score
	return center.astype(int)

# calculate accumulate intensity profile of SPI patterns
def inten_profile_vaccurate(dataset, mask, *exp_param):
	if len(exp_param)<4:
		raise ValueError("Please be sure to give all exp_param ! Exit")
	qinfo = spi_q.cal_q(exp_param[0], exp_param[1], exp_param[2], exp_param[3])
	data = dataset
	num = data.shape[0]
	size = data.shape[1:]
	intens = []
	rofq = np.inf

	for ind,pat in enumerate(data):
		center = friedel_search(pat, [size[0]//2, size[1]//2], mask)
		intens_one = radp.radial_profile(pat, center, mask)
		intens.append(intens_one[:,1])
		rofq = min(rofq, len(intens_one[:,1]))
		sys.stdout.write(str(ind)+'/'+str(num)+' patterns\r')
		sys.stdout.flush()
	final = np.zeros(rofq)
	for p in intens:
		final += p[0:rofq]
	final = final/float(num)
	return np.vstack((qinfo,final[:len(qinfo)])).T


def inten_profile_vfast(dataset, mask, *exp_param):
	if len(exp_param)<4:
		raise ValueError("Please be sure to give all exp_param ! Exit")
	import scipy.ndimage.interpolation as ndint

	qinfo = spi_q.cal_q(exp_param[0], exp_param[1], exp_param[2]*4, exp_param[3]/4.0)
	saxs = cal_saxs(dataset)
	center = friedel_search(saxs, [saxs.shape[0]//2, saxs.shape[1]//2], mask)
	newsaxs = newpat = ndint.zoom(saxs, 4)
	if mask is not None:
		newmask = np.round(ndint.zoom(mask, 4)).astype(int)
	else:
		newmask = None
	newcenter = center * 4
	intens = radp.radial_profile(newsaxs, newcenter, newmask)
	return np.vstack((qinfo,intens[0:len(qinfo),1])).T

# calculate the saxs pattern of an SPI data set
def cal_saxs(data):
	return np.sum(data, axis=0)/float(len(data))


def centering(pat, estimated_center, mask=None, small_r=None, large_r=None):
	center = friedel_search(pat, estimated_center, mask, large_r, small_r)
	ori_shape = pat.shape
	c2left = center[1]
	c2right = ori_shape[1] - center[1] - 1
	c2top = center[0]
	c2bottom = ori_shape[0] - center[0] - 1
	if c2right>c2left:
		left = 0
		right = center[1] + c2left + 1
	else:
		right = ori_shape[1]
		left = center[1] - c2right
	if c2top>c2bottom:
		bottom = ori_shape[0]
		top = center[0] - c2bottom
	else:
		top = 0
		bottom = center[0] + c2top + 1
	err = (bottom - top) - (right - left)
	if err>0:
		bottom = bottom - err//2
		top = top + err//2
	elif err<0:
		right = right + err//2
		left = left - err//2
	else:
		pass
	top = int(top)
	bottom = int(bottom)
	left = int(left)
	right = int(right)
	if mask is None:
		return pat[top:bottom,left:right]
	else:
		return pat[top:bottom,left:right], mask[top:bottom,left:right]


def particle_size(saxs, estimated_center, exparam=None, high_filter_cut=0.3, power=0.7, mask=None):
	from scipy.signal import argrelextrema
	# high pass filter
	csaxs, cmask = centering(saxs, estimated_center, mask)
	center = (np.array(csaxs.shape)-1)/2.0
	Iq = radp.radial_profile(csaxs, center, cmask)[:,1]
	cut = Iq.max()*high_filter_cut
	width = np.where(Iq>cut)[0][-1]
	x, y = grid(csaxs)
	r = np.sqrt((x-center[0])**2 + (y-center[1])**2)
	filter_s = 1 - np.exp(-(r/width)**16)
	saxs_filtered = np.abs(np.fft.ifft2( np.fft.ifftshift(np.fft.fftshift(np.fft.fft2(csaxs)) * filter_s) ))
	# auto correlation
	auto_coor = np.abs(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(saxs_filtered**power))))
	# detect particle size
	auto_coor_center = np.where(auto_coor==auto_coor.max())
	radp_auto_coor = radp.radial_profile(auto_coor, auto_coor_center)[:,1]
	# find peak
	"""
	derive = (radp_auto_coor[1:] - radp_auto_coor[:-1])/radp_auto_coor[:-1]
	peak = np.argmax(derive) + 1
	"""
	peak = argrelextrema(radp_auto_coor, np.greater, order=1)[0][0]
	if type(exparam)==str:
		try:
			param = np.array(exparam.split(',')).astype(float)
			#q_corner = cal_q(param[0], param[1], len(radp_auto_coor), param[2])[-1]
			qinfo = spi_q.cal_q(param[0], param[1], min(auto_coor.shape), param[2])
			peak = peak * 1.0/qinfo[-1]
			sizes = 1.0/qinfo[-1] * np.arange(len(radp_auto_coor))
		except:
			raise RuntimeError('given exparam is invalid')	
	else:
		sizes = np.arange(len(radp_auto_coor))
	return [peak, np.vstack([sizes, radp_auto_coor]).T]


def particle_size_sp(dataset, exparam, fitarea, badsearchr, method, mask=None, center=None, verbose=True):
	from scipy.signal import argrelextrema
	import scipy.optimize as optimization

	def func(q, R):
		qR = q*R
		return 9*(np.sin(qR)-qR*np.cos(qR))**2/qR**6

	# local var
	nr = int(fitarea[0])
	nR = int(fitarea[1])
	if nR-nr<10:
		raise ValueError("Fit area is too small.")
	detd = exparam[0]
	lamda = exparam[1]
	pixr = exparam[2]
	D = np.zeros(len(dataset))
	size = np.array(dataset.shape[1:], dtype=int)
	qinfo = 2*np.pi*spi_q.cal_q(detd, lamda, np.min(size//2), pixr)

	for i,d in enumerate(dataset):
		bad = 0
		if center is None:
			thiscenter = friedel_search(d, size//2, mask, 10, np.min([50, np.min(size//2)]))
		else:
			thiscenter = center
		Iq = radp.radial_profile(d, thiscenter, mask)[nr:nR,1]
		q_min_index = argrelextrema(Iq, np.less, order=1)[0]
		q_max_index = argrelextrema(Iq, np.greater, order=1)[0]
		q_min_index = np.sort(q_min_index)
		iq = Iq[q_max_index]
		if len(q_max_index)>5:
			if np.sum(np.argsort(iq[0:5])-np.arange(5)) != 0:
				bad = 1
		else:
			if np.sum(np.argsort(iq)-np.arange(len(iq))) != 0:
				bad = 1
		if bad == 1 \
		or len(np.where(q_min_index+nr < badsearchr)[0]) > 10 \
		or len(np.where(q_max_index+nr < badsearchr)[0]) < 2:
			print("[failed] data index is %d : q0 index is %d\n" % (i, q_min_index[0]+nr))
			D[i] = 0
			continue
		if method == "q0":
			D[i] = 2*4.493 / qinfo[ q_min_index[0]+nr ]
		else:
			R0 = 2*4.493 / qinfo[ q_min_index[0]+nr ]
			opt,cov = optimization.curve_fit(func, qinfo[nr:nR], Iq, R0)
			D[i] = opt[0]

		if verbose:
			if i%(int(np.ceil(len(dataset)/10.0))) == 0 or i==(len(dataset)-1):
				sys.stdout.write("Processing progess : %.1f\r" % (100*float(i+1)/len(dataset)))
				sys.stdout.flush()

	return D


