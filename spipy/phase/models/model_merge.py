import numpy as np
from itertools import product

def T_fourier(shape, T, is_fft_shifted = True):
    """
    e - 2pi i r q
    e - 2pi i dx n m / N dx
    e - 2pi i n m / N 
    """
    # make i, j, k for each pixel
    i = np.fft.fftfreq(shape[0])
    j = np.fft.fftfreq(shape[1])
    if len(shape) == 3:
        k = np.fft.fftfreq(shape[2])
        i, j, k = np.meshgrid(i, j, k, indexing='ij')
    else:
        i, j = np.meshgrid(i, j, indexing='ij')

    if is_fft_shifted is False :
        i = np.fft.ifftshift(i)
        j = np.fft.ifftshift(j)
        if len(shape) == 3:
            k = np.fft.ifftshift(k)

    tmp = i * T[0] + j * T[1]
    if len(shape) == 3:
        tmp += k * T[2]

    phase_ramp = np.exp(- 2J * np.pi * tmp)
    return phase_ramp

def centre(O):
    import scipy.ndimage
    a  = (O * O.conj()).real
    a  = np.fft.fftshift(a)

    aroll = []
    for i in range(len(a.shape)):
        axes = list(range(len(a.shape)))
        axes.pop(i)
        t = np.sum(a, axis = tuple(axes))
        
        dcm = [scipy.ndimage.measurements.center_of_mass(np.roll(t, i+1))[0] - \
               scipy.ndimage.measurements.center_of_mass(np.roll(t, i  ))[0]   \
               for i in range(t.shape[0])]
        
        dcm = scipy.ndimage.gaussian_filter1d(dcm, t.shape[0]/3., mode='wrap')
        
        aroll.append(np.argmax(dcm))
    
    # roughly centre O
    O = multiroll(O, aroll)
    O = np.fft.fftshift(O)

    cm = np.rint(scipy.ndimage.measurements.center_of_mass( (O*O.conj()).real)).astype(np.int)
    O  = multiroll(O, -cm)
    
    # roughly centre O
    #O = multiroll(O, aroll)
    
    # this doesn't really work (lot's of smearing from the fourier interpolation)
    # fourier shift to the centre of mass
    #O = np.fft.ifftshift(O)
    #cm = scipy.ndimage.measurements.center_of_mass((O * O.conj()).real)
    #print cm, aroll
    #O  = roll(O, cm)
    return O

def merge_sols(Os, silent=False):
    """
    grab the solutions, align the phases, un-flip and centre
    then average them.

    Also return the radial average of the merged farfield 
    diffraction pattern.
    """
    if not silent : print('\n Merging solutions')
    if not silent : print(' centering...')
    for i in range(len(Os)):
        Os[i] = centre(Os[i])
        
    if not silent : print(' aligning phases...')
    if np.any(np.iscomplex(Os[0])):
        for i in range(len(Os)):
            s     = np.sum(Os[i])
            phase = np.arctan2(s.imag, s.real)
            Os[i] = Os[i] * np.exp(- 1J * phase)
            if not silent : print('\t sum(imag) after alignment:', np.sum(Os[i].imag))

    if not silent : print('\n flipping with respect to Os[0]')
    O = Os[0]
    for i in range(1, len(Os)):
        Ot  = Os[i].copy()
        Ot  = O - Ot
        Ot  = (Ot * Ot.conj()).real
        er1 = np.sum( Ot )
    
        Ot  = Os[i].copy()
        if len(Ot.shape) == 2:
            Ot  = Ot[::-1, ::-1]
            Ot  = multiroll(Ot, [1,1])
        else:
            Ot  = Ot[::-1, ::-1, ::-1]
            Ot  = multiroll(Ot, [1,1,1])
        Ot2  = O - Ot
        Ot2 = (Ot2 * Ot2.conj()).real
        er2 = np.sum( Ot2 )

        if not silent : print('')
        if not silent : print('\t error un-flipped:', er1)
        if not silent : print('\t error    flipped:', er2)
        if er2 < er1 :
            if not silent : print('\t flipping...')
            Os[i] = Ot
    
    O = np.sum(Os, axis = 0) / float(Os.shape[0])
    if np.any(np.iscomplex(Os[0])):
        if len(O.shape) == 2:
            angle = np.angle(np.fft.fftn(Os, axes=(1,2)))
        else:
            angle = np.angle(np.fft.fftn(Os, axes=(1,2,3)))
        prft  = np.mean(np.exp(1.0J * angle), axis=0)
    else :
        prft = None
    return O, prft
    
def multiroll(x, shift, axis=None):
    """Roll an array along each axis.
    Thanks to: Warren Weckesser, 
    http://stackoverflow.com/questions/30639656/numpy-roll-in-several-dimensions
    
    
    Parameters
    ----------
    x : array_like
        Array to be rolled.
    shift : sequence of int
        Number of indices by which to shift each axis.
    axis : sequence of int, optional
        The axes to be rolled.  If not given, all axes is assumed, and
        len(shift) must equal the number of dimensions of x.
    Returns
    -------
    y : numpy array, with the same type and size as x
        The rolled array.
    Notes
    -----
    The length of x along each axis must be positive.  The function
    does not handle arrays that have axes with length 0.
    See Also
    --------
    numpy.roll
    Example
    -------
    Here's a two-dimensional array:
    >>> x = np.arange(20).reshape(4,5)
    >>> x 
    array([[ 0,  1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14],
           [15, 16, 17, 18, 19]])
    Roll the first axis one step and the second axis three steps:
    >>> multiroll(x, [1, 3])
    array([[17, 18, 19, 15, 16],
           [ 2,  3,  4,  0,  1],
           [ 7,  8,  9,  5,  6],
           [12, 13, 14, 10, 11]])
    That's equivalent to:
    >>> np.roll(np.roll(x, 1, axis=0), 3, axis=1)
    array([[17, 18, 19, 15, 16],
           [ 2,  3,  4,  0,  1],
           [ 7,  8,  9,  5,  6],
           [12, 13, 14, 10, 11]])
    Not all the axes must be rolled.  The following uses
    the `axis` argument to roll just the second axis:
    >>> multiroll(x, [2], axis=[1])
    array([[ 3,  4,  0,  1,  2],
           [ 8,  9,  5,  6,  7],
           [13, 14, 10, 11, 12],
           [18, 19, 15, 16, 17]])
    which is equivalent to:
    >>> np.roll(x, 2, axis=1)
    array([[ 3,  4,  0,  1,  2],
           [ 8,  9,  5,  6,  7],
           [13, 14, 10, 11, 12],
           [18, 19, 15, 16, 17]])
    """
    x = np.asarray(x)
    if axis is None:
        if len(shift) != x.ndim:
            raise ValueError("The array has %d axes, but len(shift) is only "
                             "%d. When 'axis' is not given, a shift must be "
                             "provided for all axes." % (x.ndim, len(shift)))
        axis = range(x.ndim)
    else:
        # axis does not have to contain all the axes.  Here we append the
        # missing axes to axis, and for each missing axis, append 0 to shift.
        missing_axes = set(range(x.ndim)) - set(axis)
        num_missing = len(missing_axes)
        axis = tuple(axis) + tuple(missing_axes)
        shift = tuple(shift) + (0,)*num_missing

    # Use mod to convert all shifts to be values between 0 and the length
    # of the corresponding axis.
    shift = [s % x.shape[ax] for s, ax in zip(shift, axis)]

    # Reorder the values in shift to correspond to axes 0, 1, ..., x.ndim-1.
    shift = np.take(shift, np.argsort(axis))

    # Create the output array, and copy the shifted blocks from x to y.
    y = np.empty_like(x)
    src_slices = [(slice(n-shft, n), slice(0, n-shft))
                  for shft, n in zip(shift, x.shape)]
    dst_slices = [(slice(0, shft), slice(shft, n))
                  for shft, n in zip(shift, x.shape)]
    src_blks = product(*src_slices)
    dst_blks = product(*dst_slices)
    for src_blk, dst_blk in zip(src_blks, dst_blks):
        y[dst_blk] = x[src_blk]

    return y



if __name__ == '__main__':
    pass
