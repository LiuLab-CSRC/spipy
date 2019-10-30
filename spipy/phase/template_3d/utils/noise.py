import numpy as np

def add_noise_3d(diff, n, is_fft_shifted = True, remove_courners = True, unit_cell_size=None):
    """
    Add Poisson noise to a 3d volume.

    This function takes into account the
    reduced counting statistics at higher 
    resolutions.

    n = is the mean number of photons detected 
        in a speckle at the middle edge of the 
        detector. Assuming oversampling of 2
        in each direction.
    
    Expected number of photons for voxel R:
    = I(R) 3u [(R+u)^2 - R^2] / [(R+u)^3 - R^3]
    
    for a square pixel of side length u. This 
    depends somewhat on the merging strategy.
    To first order in u this becomes:
    = I(R) 2u / R
    
    This is not valid for the courners of the 
    detector that are sampled much less.
    """
    if is_fft_shifted is False :
        diff_out = np.fft.ifftshift(diff.copy())
    else :
        diff_out = diff.copy()
    
    norm = np.sum(diff)
    
    mask = np.ones_like(diff, dtype = np.bool)
    
    i = np.fft.fftfreq(diff.shape[0]) * diff.shape[0]
    j = np.fft.fftfreq(diff.shape[1]) * diff.shape[1]
    k = np.fft.fftfreq(diff.shape[2]) * diff.shape[2]
    i, j, k = np.meshgrid(i, j, k, indexing='ij')
    R      = np.sqrt(i**2 + j**2 + k**2)
    R[0, 0, 0] = 0.5
    
    # R scaling
    R_scale   = 3. * ((R+1.)**2 - R**2) / ((R+1.)**3 - R**3)
    diff_out  = diff_out * R_scale

    # normalise
    diff_out = diff_out / np.sum(diff_out)

    # calculate the total number of photons
    # from the mean number of photons per speckle
    # at the edge of the detector
    if unit_cell_size is not None :
        # ratio of diff vol to unit_cell vol
        over_sampling = float(diff.size) / float(unit_cell_size**3) 
    else :
        over_sampling = 2.

    rav = rad_av(diff_out)
    N = float(n) / (over_sampling * rav[int(np.min(diff.shape) / 2. - 1.)])
    print('total number of photons required:', int(N))
    print('oversampling :', over_sampling)

    # Poisson sampling
    diff_out = np.random.poisson(lam = N * diff_out).astype(np.float64)

    # un-scale
    #R_scale  /= np.mean(R_scale) 
    diff_out /= R_scale

    # renormalise
    diff_out = diff_out * norm / np.sum(diff_out)

    if remove_courners :
        l = np.where(R >= np.min(diff.shape) / 2.)
        diff_out[l] = 0.0
        mask[l]     = False

    return diff_out, mask

def rad_av(diff, rs = None, is_fft_shifted = True):
    if rs is None :
        i = np.fft.fftfreq(diff.shape[0]) * diff.shape[0]
        j = np.fft.fftfreq(diff.shape[1]) * diff.shape[1]
        k = np.fft.fftfreq(diff.shape[2]) * diff.shape[2]
        i, j, k = np.meshgrid(i, j, k, indexing='ij')
        rs      = np.sqrt(i**2 + j**2 + k**2).astype(np.int16).ravel()
        
        if is_fft_shifted is False :
            rs = np.fft.fftshift(rs)
    
    ########### Find the radial average
    # get the r histogram
    r_hist = np.bincount(rs)
    # get the radial total 
    r_av = np.bincount(rs, diff.ravel())
    # prevent divide by zero
    nonzero = np.where(r_hist != 0)
    # get the average
    r_av[nonzero] = r_av[nonzero] / r_hist[nonzero].astype(r_av.dtype)
    return r_av
