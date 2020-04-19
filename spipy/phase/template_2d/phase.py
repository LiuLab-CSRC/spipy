import numpy as np
import sys, os
import re
import copy

from src import era, raar, dm
from utils import io_utils, merge

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def config_iters_to_alg_num(string):
    # split a string like '100ERA 200DM 50ERA' with the numbers
    steps = re.split(r'(\d+)', string)   # ['', '100', 'ERA ', '200', 'DM ', '50', 'ERA']
    
    # get rid of empty strings
    steps = [s for s in steps if len(s)>0] # ['100', 'ERA ', '200', 'DM ', '50', 'ERA']
    
    # pair alg and iters
    # [['ERA', 100], ['DM', 200], ['ERA', 50]]
    alg_iters = [ [steps[i+1].strip(), int(steps[i])] for i in range(0, len(steps), 2)]
    return alg_iters

def out_merge(out, I, good_pix):
    # average the background retrievals
    if out[0]['background'] is not None :
        background = np.mean([i['background'] for i in out], axis=0)
    else :
        background = 0

    silent = True
    if rank == 0: silent = False
    
    # centre, flip and average the retrievals
    O, PRTF    = merge.merge_sols(np.array([i['O'] for i in out]), True)
    support, t = merge.merge_sols(np.array([i['support'] for i in out]).astype(np.float), True)
       
    eMod    = np.array([i['eMod'] for i in out])
    eCon    = np.array([i['eCon'] for i in out])

    # mpi
    comm.Barrier()
    
    O          = comm.gather(O, root=0)
    support    = comm.gather(support, root=0)
    eMod       = comm.gather(eMod, root=0)
    eCon       = comm.gather(eCon, root=0)
    PRTF       = comm.gather(PRTF, root=0)
    if background is not 0 :
        background = comm.gather(background, root=0)
    
    if rank == 0 :
        PRTF           = np.abs(np.mean(np.array(PRTF), axis=0))
        #t, t, PRTF_rav = phasing_3d.src.era.radial_symetry(PRTF)
        
        eMod       = np.array(eMod).reshape((size*eMod[0].shape[0], eMod[0].shape[1]))
        eCon       = np.array(eCon).reshape((size*eCon[0].shape[0], eCon[0].shape[1]))
        O, t       = merge.merge_sols(np.array(O))
        support, t = merge.merge_sols(np.array(support))
        if background is not 0 :
            background = np.mean(np.array(background), axis=0)
        
    if rank == 0 :
        # get the prft
        #PRTF, PRTF_rav = utils.merge.PRTF(O, I, background, good_pix)

        # get the PSD
        PSD, PSD_I, PSD_phase = merge.PSD(O, I)

        out_m = out[0]
        out_m['I'] = np.abs(np.fft.fftn(O))**2
        out_m['O'] = O
        out_m['background'] = background
        out_m['PSD']      = PSD
        out_m['PSD_I']    = PSD_I
        out_m['PRTF']     = PRTF
        out_m['PRTF_rav'] = np.array([0]) #PRTF_rav
        out_m['eMod']     = eMod
        out_m['eCon']     = eCon
        out_m['support']  = support
        return out_m
    else :
        return None
    


def phase(I, support, params, good_pix = None, sample_known = None):
    d   = {'eMod' : [],         \
           'eCon' : [],         \
           'O'    : None,       \
           'background' : None, \
           'B_rav' : None, \
           'support' : None     \
            }
    out = []

    if sample_known is None:
        params['phasing_parameters']['O'] = None
    else:
        c_dtype = (I[0,0] + 1J * I[0,0]).dtype
        params['phasing_parameters']['O'] = np.array(sample_known).astype(c_dtype)
    
    params['phasing_parameters']['mask'] = good_pix
    
    if params['phasing_parameters']['support'] is None :
        params['phasing_parameters']['support'] = support

    params0 = copy.deepcopy(params)
    
    alg_iters = config_iters_to_alg_num(params['phasing']['iters'])
        
    # Repeats
    #---------------------------------------------
    for j in range(params['phasing']['repeats']):
        out.append(copy.deepcopy(d))
        params = copy.deepcopy(params0)
        
        for alg, iters in alg_iters :
            
            if alg == 'ERA':
                O, info = era.ERA(I, iters, **params['phasing_parameters'])
             
            elif alg == 'DM':
                O, info = dm.DM(I,  iters, **params['phasing_parameters'])

            elif alg == 'RAAR':
                O, info = raar.RAAR(I, iters, **params['phasing_parameters'])

            else:
                raise RuntimeError("Input algorithms '%s' is invalid !" % alg)
                MPI.Finalize()
             
            out[j]['O']           = params['phasing_parameters']['O']          = O
            out[j]['eMod']       += info['eMod']
            out[j]['eCon']       += info['eCon']
            
            if 'background' in info.keys():
                out[j]['background']  = params['phasing_parameters']['background'] = info['background'] * good_pix
                out[j]['B_rav']       = info['r_av']
    
        out[j]['support']     = params['phasing_parameters']['support']    = info['support']
    return out



if __name__ == "__main__":
    args = io_utils.parse_cmdline_args_phasing()
    
    # read the h5 file
    diff, support, good_pix, sample_known, params = io_utils.read_input_h5(args.input)

    out = phase(diff, support, params, \
                        good_pix = good_pix, sample_known = sample_known)

    out = out_merge(out, diff, good_pix)
    
    # write the h5 file 
    if rank == 0 :
        io_utils.write_output_h5(params['output']['path'], diff, out['I'], support, out['support'], \
                                      good_pix, sample_known, out['O'], out['eMod'], out['eCon'], None,   \
                                      out['PRTF'], out['PRTF_rav'], out['PSD'], out['PSD_I'], out['B_rav'])

