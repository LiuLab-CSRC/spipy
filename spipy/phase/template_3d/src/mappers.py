import numpy as np
import sys
import copy

def isValid(thing, d=None):
    """
    checks if 'thing' is valid. If d (a dictionary is not None) then
    check if d['thing'] is valid.
    """
    valid = False
    
    if d is not None :
        if thing not in d.keys():
            return valid 
        else :
            thing2 = d[thing]
    
    if thing2 is not None and thing2 is not False :
        valid = True
    
    return valid

class Modes(dict):
    def __init__(self, **args):
        dict.__init__(self, **args)

    def __add__(self, value):
        out = self.copy()
        for k in self.keys():
            if type(value) == Modes :
                out[k] = self[k] + value[k]
            else :
                out[k] = self[k] + value
        return out

    def __iadd__(self, value):
        for k in self.keys():
            if type(value) == Modes :
                self[k] += value[k]
            else :
                self[k] += value
        return self

    def __sub__(self, value):
        out = self.copy()
        for k in self.keys():
            if type(value) == Modes :
                out[k] = self[k] - value[k]
            else :
                out[k] = self[k] - value
        return out

    def __isub__(self, value):
        for k in self.keys():
            if type(value) == Modes :
                self[k] -= value[k]
            else :
                self[k] -= value
        return self

    def __mul__(self, value):
        out = self.copy()
        for k in self.keys():
            if type(value) == Modes :
                out[k] = self[k] * value[k]
            else :
                out[k] = self[k] * value
        return out
    
    def __imul__(self, value):
        for k in self.keys():
            if type(value) == Modes :
                self[k] *= value[k]
            else :
                self[k] *= value
        return self

    def copy(self):
        out = Modes()
        for k in self.keys():
            out[k] = self[k].copy()
        return out

class Mapper():
    
    def __init__(self, I, **args):
        modes = Modes()
        
        # check if there is a background
        if isValid('background', args):
            if args['background'] is True :
                modes['B'] = np.random.random((I.shape)).astype(args['dtype'])
            else :
                modes['B'] = np.sqrt(args['background']).astype(args['dtype'])

        if isValid('O', args):
            modes['O'] = args['O']
        else :
            modes['O'] = np.random.random(I.shape).astype(args['c_dtype'])
        
        # this is the radial value for every pixel 
        # in the volume
        self.rs    = None 
        
        self.mask = 1
        if isValid('mask', args):
            self.mask = args['mask']
        
        self.alpha = 1.0e-10
        if isValid('alpha', args):
            self.alpha = args['alpha']

        self.I_norm = (self.mask * I).sum()
        self.amp   = np.sqrt(I.astype(args['dtype']))

        # define the data projection
        # --------------------------
        if 'B' in modes.keys() :
            self.Pmod = self.Pmod_back
        else :
            self.Pmod = self.Pmod_single
    
        # define the support projection
        # -----------------------------
        if isValid('voxel_number', args) :
            self.voxel_number = args['voxel_number']
        else :
            self.voxel_number = False
            self.S    = args['support']
        
        self.support = None
        if isValid('support', args):
            self.support = args['support']

        self.modes = modes

    def object(self, modes):
        return modes['O']

    def Psup(self, modes):
        out = modes.copy()

        if self.voxel_number :
            O = out['O']
            self.S = choose_N_highest_pixels( (O * O.conj()).real, self.voxel_number, support = self.support)

        out['O'] *= self.S

        if 'B' in modes.keys() :
            out['B'], self.rs, self.r_av = radial_symetry(out['B'], rs = self.rs)
        return out

    def Pmod_single(self, modes):
        out = modes.copy()
        out['O'] = pmod_single(self.amp, modes['O'], self.mask, alpha = self.alpha)
        return out
    
    def Pmod_back(self, modes):
        out = modes.copy()
        out['O'], out['B'] = pmod_back(self.amp, modes['B'], modes['O'], self.mask, alpha = self.alpha)
        return out

    def Imap(self, modes):
        O = np.fft.fftn(modes['O']) 
        if 'B' in modes.keys() :
            I = (O.conj() * O).real + modes['B']**2
        else :
            I = (O.conj() * O).real 
        return I
    
    def Emod(self, modes):
        M         = self.Imap(modes)
        eMod      = np.sum( self.mask * ( np.sqrt(M) - self.amp )**2 )
        eMod      = np.sqrt( eMod / self.I_norm )
        return eMod

    def finish(self, modes):
        out = {}
        out['support'] = self.S
        out['I']       = self.Imap(modes)

        if 'B' in modes.keys() :
            out['background'] = modes['B']**2
            out['r_av']       = self.r_av
        return out

    def l2norm(self, delta, array0):
        num = 0
        den = 0
        for k in delta.keys():
            num += np.sum( (delta[k] * delta[k].conj()).real ) 
            den += np.sum( (array0[k] * array0[k].conj()).real ) 
        return np.sqrt(num / den)

def choose_N_highest_pixels_slow(array, N):
    percent = (1. - float(N) / float(array.size)) * 100.
    thresh  = np.percentile(array, percent)
    support = array > thresh
    # print '\n\nchoose_N_highest_pixels'
    # print 'percentile         :', percent, '%'
    # print 'intensity threshold:', thresh
    # print 'number of pixels in support:', np.sum(support)
    return support

def choose_N_highest_pixels(array, N, tol = 1.0e-5, maxIters=1000, support=None):
    """
    Use bisection to find the root of
    e(x) = \sum_i (array_i > x) - N

    then return (array_i > x) a boolean mask

    This is faster than using percentile (surprising)

    If support is not None then values outside the support
    are ignored. 
    """
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

def pmod_single(amp, O, mask = 1, alpha = 1.0e-10):
    O = np.fft.fftn(O)
    O = Pmod_single(amp, O, mask = mask, alpha = alpha)
    O = np.fft.ifftn(O)
    return O
    
def Pmod_single(amp, O, mask = 1, alpha = 1.0e-10):
    out  = mask * O * amp / (abs(O) + alpha)
    out += (1 - mask) * O
    return out

def pmod_back(amp, background, O, mask = 1, alpha = 1.0e-10):
    O = np.fft.fftn(O)
    O, background = Pmod_back(amp, background, O, mask = mask, alpha = alpha)
    O = np.fft.ifftn(O)
    return O, background
    
def Pmod_back(amp, background, O, mask = 1, alpha = 1.0e-10):
    M = mask * amp / np.sqrt((O.conj() * O).real + background**2 + alpha)
    out         = O * M
    background *= M
    out += (1 - mask) * O
    return out, background

def radial_symetry(background, rs = None, is_fft_shifted = True):
    if rs is None :
        i = np.fft.fftfreq(background.shape[0]) * background.shape[0]
        j = np.fft.fftfreq(background.shape[1]) * background.shape[1]
        k = np.fft.fftfreq(background.shape[2]) * background.shape[2]
        i, j, k = np.meshgrid(i, j, k, indexing='ij')
        rs      = np.sqrt(i**2 + j**2 + k**2).astype(np.int16)
        
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
