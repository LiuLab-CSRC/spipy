import numpy as np

def zero_pad_to_nearest_pow2(diff, shape_new = None):
    """
    find the smallest power of 2 that 
    fits each dimension of the diffraction
    pattern then zero pad keeping the zero
    pixel centred
    """
    if shape_new is None :
        shape_new = []
        for s in diff.shape:
            n = 0
            while 2**n < s :
                n += 1
            shape_new.append(2**n)

    print('\n reshaping:', diff.shape, '-->', shape_new)
    diff_new = np.zeros(tuple(shape_new), dtype=diff.dtype)
    diff_new[:diff.shape[0], :diff.shape[1]] = diff

    # roll the axis to keep N / 2 at N'/2
    for i in range(len(shape_new)):
        diff_new = np.roll(diff_new, shape_new[i]//2 - diff.shape[i]//2, i)
    return diff_new

def mk_circle(shape, rad):
    i, j = np.indices(shape)
    r       = (i-shape[0]//2)**2 + (j-shape[1]//2)**2
    circle  = r < rad**2
    return circle

def mk_gaus(shape, sigma):
    i, j = np.indices(shape)
    r       = (i-shape[0]//2)**2 + (j-shape[1]//2)**2
    gaus    = np.exp(-0.5 * r / sigma**2) / (sigma**2 * 2. * np.pi)**1.5
    return gaus

def mk_Fgaus(shape, sigma):
    i, j = np.indices(shape)
    i = (i.astype(np.float64) - shape[0]//2)/ float(shape[0])
    j = (j.astype(np.float64) - shape[1]//2)/ float(shape[1])
    r       = i**2 + j**2
    gaus    = np.exp(-2. * r * (np.pi * sigma)**2)
    return gaus
