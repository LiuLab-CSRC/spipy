import numpy as np
import sys

from .mappers import Mapper
from .mappers import isValid

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def ERA(I, iters, **args):
    """
    Find the phases of 'I' given O using the Error Reduction Algorithm.
    
    Parameters
    ----------
    I : numpy.ndarray, (N, M)
        Merged diffraction patterns to be phased. 
    
        N : the number of pixels along slowest scan axis of the detector
        M : the number of pixels along slow scan axis of the detector
        K : the number of pixels along fast scan axis of the detector
    
    iters : int
        The number of ERA iterations to perform.
    
    O : numpy.ndarray, (N, M) 
        The real-space scattering density of the object such that:
            I = |F[O]|^2
        where F[.] is the 3D Fourier transform of '.'.     
    
    support : (numpy.ndarray, None or int), (N, M)
        Real-space region where the object function is known to be zero. 
        If support is an integer then the N most intense pixels will be kept at
        each iteration.
    
    mask : numpy.ndarray, (N, M), optional, default (1)
        The valid detector pixels. Mask[i, j] = 1 (or True) when the detector pixel 
        i, j, k is valid, Mask[i, j] = 0 (or False) otherwise.
    
    hardware : ('cpu', 'gpu'), optional, default ('cpu') 
        Choose to run the reconstruction on a single cpu core ('cpu') or a single gpu
        ('gpu'). The numerical results should be identical.
    
    alpha : float, optional, default (1.0e-10)
        A floating point number to regularise array division (prevents 1/0 errors).
    
    dtype : (None, 'single' or 'double'), optional, default ('single')
        Determines the numerical precision of the calculation. If dtype==None, then
        it is determined from the datatype of I.

    Mapper : class, optional, default None
        A mapping class that provides the methods supplied by:
            phasing_3d.src.mappers.Mapper
    
    Returns
    -------
    O : numpy.ndarray, (U, V) 
        The real-space object function after 'iters' iterations of the ERA algorithm.
    
    info : dict
        contains diagnostics:
            
            'I'     : the diffraction pattern corresponding to object above
            'eMod'  : the modulus error for each iteration:
                      eMod_i = sqrt( sum(| O_i - Pmod(O_i) |^2) / I )
            'eCon'  : the convergence error for each iteration:
                      eCon_i = sqrt( sum(| O_i - O_i-1 |^2) / sum(| O_i |^2) )
        
    Notes 
    -----
    The ERA is the simplest iterative projection algorithm. It proceeds by 
    progressive projections of the exit surface waves onto the set of function that 
    satisfy the:
        modulus constraint : after propagation to the detector the exit surface waves
                             must have the same modulus (square root of the intensity) 
                             as the detected diffraction patterns (the I's).
        
        support constraint : the exit surface waves (W) must be separable into some object 
                                 and probe functions so that W_n = O_n x P.
    
    The 'projection' operation onto one of these constraints makes the smallest change to the set 
    of exit surface waves (in the Euclidean sense) that is required to satisfy said constraint.
    Examples 
    --------
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

    # Set the Mapper for the single mode (default)
    # ---------------------------------------
    # this guy is responsible for doing:
    #   I     = mapper.Imap(modes)   # mapping the modes to the intensity
    #   modes = mapper.Pmod(modes)   # applying the data projection to the modes
    #   modes = mapper.Psup(modes)   # applying the support projection to the modes
    #   O     = mapper.object(modes) # the main object of interest
    #   dict  = mapper.finish(modes) # add any additional output to the info dict
    # ---------------------------------------
    mapper = Mapper(I, **args)
    modes  = mapper.modes

    if iters > 0 and rank == 0 :
        print('\n\nalgrithm progress iteration convergence modulus error')
    
    for i in range(iters) :
        modes0 = modes.copy()
        
        # modulus projection 
        # ------------------
        modes = mapper.Pmod(modes)

        modes1 = modes.copy()
        
        # support projection 
        # ------------------
        modes = mapper.Psup(modes)
        
        # metrics
        #modes1 -= modes0
        #eMod    = mapper.l2norm(modes1, modes0)
        eMod    = mapper.Emod(modes)
        
        modes0 -= modes
        eCon    = mapper.l2norm(modes0, modes)
        
        if rank == 0 : update_progress(i / max(1.0, float(iters-1)), 'ERA', i, eCon, eMod )
        
        eMods.append(eMod)
        eCons.append(eCon)
    
    info = {}
    info['eMod']  = eMods
    info['eCon']  = eCons
    
    info.update(mapper.finish(modes))
    
    O = mapper.object(modes)
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
    sys.stdout.flush()

    
