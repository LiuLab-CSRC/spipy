import numpy as np
import sys

from .mappers import Mapper
from .mappers import isValid

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def DM(I, iters, **args):
    """
    Find the phases of 'I' given O using the Error Reduction Algorithm.
    
    Parameters
    ----------
    I : numpy.ndarray, (N, M)
        Merged diffraction patterns to be phased. 
    
        N : the number of pixels along slowest scan axis of the detector
        M : the number of pixels along slow scan axis of the detector
        K : the number of pixels along fast scan axis of the detector
    
    O : numpy.ndarray, (N, M) 
        The real-space scattering density of the object such that:
            I = |F[O]|^2
        where F[.] is the 3D Fourier transform of '.'.     
    
    iters : int
        The number of ERA iterations to perform.
    
    support : numpy.ndarray or None, (N, M)
        Real-space region where the object function is known to be zero. 
    
    mask : numpy.ndarray, (N, M), optional, default (1)
        The valid detector pixels. Mask[i, j] = 1 (or True) when the detector pixel 
        i, j, k is valid, Mask[i, j] = 0 (or False) otherwise.
    
    method : (None, 1, 2, 3, 4), optional, default (None)
        method = None :
            Automattically choose method 1, 2 based on the contents of 'background'.
            if   'background' == None then method = 1
            elif 'background' != None then method = 2
        method = 1 :
            Just update 'O'
        method = 2 :
            Update 'O' and 'background'
    
    hardware : ('cpu', 'gpu'), optional, default ('cpu') 
        Choose to run the reconstruction on a single cpu core ('cpu') or a single gpu
        ('gpu'). The numerical results should be identical.
    
    alpha : float, optional, default (1.0e-10)
        A floating point number to regularise array division (prevents 1/0 errors).
    
    dtype : (None, 'single' or 'double'), optional, default ('single')
        Determines the numerical precision of the calculation. If dtype==None, then
        it is determined from the datatype of I.
    
    full_output : bool, optional, default (True)
        If true then return a bunch of diagnostics (see info) as a python dictionary 
        (a list of key : value pairs).
    
    Returns
    -------
    O : numpy.ndarray, (U, V) 
        The real-space object function after 'iters' iterations of the ERA algorithm.
    
    info : dict, optional
        contains diagnostics:
            
            'I'     : the diffraction pattern corresponding to object above
            'eMod'  : the modulus error for each iteration:
                      eMod_i = sqrt( sum(| O_i - Pmod(O_i) |^2) / I )
            'eCon'  : the convergence error for each iteration:
                      eCon_i = sqrt( sum(| O_i - O_i-1 |^2) / sum(| O_i |^2) )
        
    Notes 
    -----
    The Difference Map algorithm [1] applies the modulus and consistency constraints
    in wierd and wonderful ways. Unlike the ERA, no iterate of DM ever fully satisfies 
    either constraint. Instead it tries to find the solution by avoiding stagnation
    points (a typical problem with ERA). It is recommended to combine DM and ERA when
    phasing. The modulus and consistency constraints are:
        modulus constraint : after propagation to the detector the exit surface waves
                             must have the same modulus (square root of the intensity) 
                             as the detected diffraction patterns (the I's).
        
        support constraint : the exit surface waves (W) must be separable into some object 
                                 and probe functions so that W_n = O_n x P.
    
    The 'projection' operation onto one of these constraints makes the smallest change to the 
    set of exit surface waves (in the Euclidean sense) that is required to satisfy said 
    constraint.
    DM applies the following recursion on the state vector:
        f_i+1 = f_i + b (Ps Rm f_i - Pm Rs f_i)
    where
        Rs f = ((1 + gm)Ps - gm)f
        Rm f = ((1 + gs)Pm - gs)f
    and f_i is the i'th iterate of DM (in our case the exit surface waves). 'gs', 'gm' and 
    'b' are real scalar parameters. gs and gm are the degree of relaxation for the consistency
    and modulus constraint respectively. While |b| < 1 can be thought of as relating to 
    step-size of the algorithm. Once DM has reached a fixed point, so that f_i+1 = f_i, 
    the solution (f*) is obtained from either of:
        f* = Ps Rm f_i = PM Rs f_i
    
    One choice for gs and gm is to set gs = -1/b, gm = 1/b and b=1 leading to:
        Rs f  = 2 Ps f - f
        Rm f  = f
        f_i+1 = f_i + Ps f_i - Pm (2 Ps f_i - f_i)
        and 
        f* = Ps f_i = PM (2 Ps f_i - f_i)
    
    Examples 
    --------
    References
    ----------
    [1] Veit Elser, "Phase retrieval by iterated projections," J. Opt. Soc. Am. A 
        20, 40-55 (2003)
    """
    """
    if hardware == 'gpu':
        from dm_gpu import DM_gpu
        return DM_gpu(I, R, P, O, iters, OP_iters, mask, background, method, hardware, alpha, dtype, full_output)
    """
    # set the real and complex data precision
    # ---------------------------------------
    if 'dtype' not in args.keys() :
        dtype   = I.dtype
        c_dtype = (I[0,0] + 1J * I[0, 0]).dtype
    
    elif args['dtype'] == 'single':
        dtype   = np.float32
        c_dtype = np.complex64

    elif args['dtype'] == 'double':
        dtype   = np.float64
        c_dtype = np.complex128

    args['dtype']   = dtype
    args['c_dtype'] = c_dtype

    if isValid('Mapper', args) : 
        if rank == 0 : print('\nusing user defined mapper')
        Mapper = args['Mapper']
    else :
        if rank == 0 : print('\nusing default cpu mapper')
        from .mappers import Mapper 
    
    eMods     = []
    eCons     = []
    
    mapper = Mapper(I, **args)
    modes  = mapper.modes
    
    modes_sup = mapper.Psup(modes)

    if iters > 0  and rank==0:
        print('\n\nalgrithm progress iteration convergence modulus error')
    
    for i in range(iters) :
        
        # reference
        modes0 = modes.copy()
        
        # update 
        #-------
        modes += mapper.Pmod(modes_sup * 2 - modes) - modes_sup
        
        # metrics
        #--------
        modes0 -= modes
        eCon    = mapper.l2norm(modes0, modes)

        # f* = Ps f_i = PM (2 Ps f_i - f_i)
        modes_sup = mapper.Psup(modes)

        eMod = mapper.Emod(modes_sup)
        
        if rank == 0 : update_progress(i / max(1.0, float(iters-1)), 'DM', i, eCon, eMod )
        
        eMods.append(eMod)
        eCons.append(eCon)
    
    info = {}
    info['eMod']  = eMods
    info['eCon']  = eCons
    
    O = mapper.object(modes_sup)
    
    info.update(mapper.finish(modes_sup))
    
    return O, info

def update_progress(progress, algorithm, i, emod, esup):
    barLength = 15 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    text = "\r{0}: [{1}] {2}% {3} {4} {5} {6} {7}".format(algorithm, "#"*block + "-"*(barLength-block), int(progress*100), i, emod, esup, status, " " * 5) # this last bit clears the line
    sys.stdout.write(text)
