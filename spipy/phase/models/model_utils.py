import numpy as np
from .model_interface import nodeData

def make_holes(shape, center, rad):
    '''
        make holes on pattern , before fftshift
        return : a bool numpy array, shape=shape
    '''
    meshgrids = np.indices(shape)
    r = np.sqrt(sum( ((grid - c)**2 for grid, c in zip(meshgrids, center)) ))
    return r < rad

def choose_N_highest_pixels(array, N, tol = 1.0e-5, maxIters=1000, support=None):
    '''
    Use bisection to find the root of
    e(x) = \sum_i (array_i > x) - N
    then return (array_i > x) a boolean mask
    This is faster than using percentile (surprising)
    If support is not None then values outside the support
    are ignored. 
    '''
    s0 = array.max()
    s1 = array.min()

    if support is not None :
        a = array[support > 0]
    else :
        a = array
        support = 1
    
    for i in range(maxIters):
        s = (s0 + s1) / 2.
        e = np.sum(a > s) - N
    
        if np.abs(e) < tol :
            break

        if e < 0 :
            s0 = s
        else :
            s1 = s
        
    S = (array > s) * support
    #print 'number of pixels in support:', np.sum(support), i, s, e
    return S


def radial_symetry(background, rs = None, is_fft_shifted = True):
    if rs is None :
        i = np.fft.fftfreq(background.shape[0]) * background.shape[0]
        j = np.fft.fftfreq(background.shape[1]) * background.shape[1]
        if len(background.shape) == 3:
            k = np.fft.fftfreq(background.shape[2]) * background.shape[2]
            i, j, k = np.meshgrid(i, j, k, indexing='ij')
            rs = np.sqrt(i**2 + j**2 + k**2).astype(np.int16)
        else:
            i, j = np.meshgrid(i, j, indexing='ij')
            rs = np.sqrt(i**2 + j**2).astype(np.int16)
        
        if is_fft_shifted is False :
            rs = np.fft.fftshift(rs)
        rs = rs.ravel()
    
    ########### Find the radial average
    # get the r histogram
    r_hist = np.bincount(rs)
    # get the radial total 
    r_av = np.bincount(rs, background.ravel())
    # prevent divide by zero
    nonzero = np.where(r_hist != 0)
    zero    = np.where(r_hist == 0)
    # get the average
    r_av[nonzero] = r_av[nonzero] / r_hist[nonzero].astype(r_av.dtype)
    r_av[zero]    = 0

    ########### Make a large background filled with the radial average
    background = r_av[rs].reshape(background.shape)
    return background, rs, r_av


def __support_projection(sample_ret, fixed_support, support_size, background = None, bg_av = None, radial_s = None):
    '''
        sample_ret is retrieved sample pattern
        fixed_support is fixed support 2d array
        support_size is the number of pixels left after shrinkwrap (the others are set to 0)
    '''
    if support_size > 0:
        Sup = choose_N_highest_pixels( np.abs(sample_ret)**2, support_size, support = fixed_support)

    sample_ret *= Sup
    if background is not None:
        background, radial_s, bg_av = radial_symetry(background, rs = radial_s)

    return sample_ret, background, radial_s, bg_av, Sup


def support_projection(Data, support_size, radial_s):
    Data.sample_ret, Data.background, radial_s, Data.background_av, Data.support = __support_projection\
        (Data.sample_ret, Data.init_support, support_size, Data.background, radial_s)
    return radial_s

########
# important
########
def support_projection_node(Datapack, node, support_size, radial_s):
    node_ = nodeData(node)
    if 'background' in node_.data.keys():
        node_.data['sample_ret'], node_.data['background'], radial_s, background_av, support = __support_projection\
        (node_.data['sample_ret'], Datapack.init_support, support_size, node_.data['background'], radial_s)
    else:
        node_.data['sample_ret'], _, radial_s, background_av, support = __support_projection\
        (node_.data['sample_ret'], Datapack.init_support, support_size, None, radial_s)
    return node_, radial_s, background_av, support


def __module_projection(diff_amp, sample_ret, background = None, good_pixel = 1, epsilon = 1e-10):
    '''
        diff_amp is diffraction amplitute, sqrt(diffraction_intensity)
        sample_ret is retrieved sample pattern, numpy 2d array
        if there is background, set it, numpy 2d array
        good_pixel : good pixel is 1
    '''
    intens_out = np.fft.fftn(sample_ret)
    if background is None:
        out  = good_pixel * intens_out * diff_amp / (np.abs(intens_out) + epsilon)
        out += (1 - good_pixel) * intens_out
    else:
        M = good_pixel * diff_amp / np.sqrt((intens_out.conj() * intens_out).real + background**2 + epsilon)
        out         = intens_out * M
        background *= M
        out += (1 - good_pixel) * intens_out
    sample_ret = np.fft.ifftn(out)

    return sample_ret, background


def module_projection(Data, epsilon = 1e-10):
    Data.sample_ret, Data.background = __module_projection(Data.diffraction_amp, Data.sample_ret, Data.background, Data.good_pixel, epsilon)

########
# important
########
def module_projection_node(Datapack, node, epsilon = 1e-10):
    node_ = nodeData(node)
    if 'background' in node_.data.keys():
        node_.data['sample_ret'], node_.data['background'] = \
            __module_projection(Datapack.diffraction_amp, node_.data['sample_ret'], \
            node_.data['background'], Datapack.good_pixel, epsilon)
    else:
        node_.data['sample_ret'], _ = \
            __module_projection(Datapack.diffraction_amp, node_.data['sample_ret'], \
            None, Datapack.good_pixel, epsilon)
    return node_