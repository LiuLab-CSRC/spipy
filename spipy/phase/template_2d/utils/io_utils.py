import numpy as np
import sys

def parse_cmdline_args():
    import argparse
    import os
    parser = argparse.ArgumentParser(prog = 'reconstruct.py', description='phase a merged 3D diffraction volume')
    parser.add_argument('config', type=str, \
                        help="file name of the configuration file")
    parser.add_argument('-i', '--input', action='store_true', \
                        help="generate the input file and exit")
    args = parser.parse_args()

    # check that args.ini exists
    if not os.path.exists(args.config):
        raise NameError('config file does not exist: ' + args.config)
    return args


def parse_parameters(config):
    """
    Parse values from the configuration file and sets internal parameter accordingly
    The parameter dictionary is made available to both the workers and the master nodes

    The parser tries to interpret an entry in the configuration file as follows:

    - If the entry starts and ends with a single quote, it is interpreted as a string
    - If the entry is the word None, without quotes, then the entry is interpreted as NoneType
    - If the entry is the word False, without quotes, then the entry is interpreted as a boolean False
    - If the entry is the word True, without quotes, then the entry is interpreted as a boolean True
    - If non of the previous options match the content of the entry, the parser tries to interpret the entry in order as:

        - An integer number
        - A float number
        - A string

      The first choice that succeeds determines the entry type
    """

    monitor_params = {}

    for sect in config.sections():
        monitor_params[sect]={}
        for op in config.options(sect):
            monitor_params[sect][op] = config.get(sect, op)
            if monitor_params[sect][op].startswith("'") and monitor_params[sect][op].endswith("'"):
                monitor_params[sect][op] = monitor_params[sect][op][1:-1]
                continue
            if monitor_params[sect][op] == 'None':
                monitor_params[sect][op] = None
                continue
            if monitor_params[sect][op] == 'False':
                monitor_params[sect][op] = False
                continue
            if monitor_params[sect][op] == 'True':
                monitor_params[sect][op] = True
                continue
            try:
                monitor_params[sect][op] = int(monitor_params[sect][op])
                continue
            except :
                try :
                    monitor_params[sect][op] = float(monitor_params[sect][op])
                    continue
                except :
                    # attempt to pass as an array of ints e.g. '1, 2, 3'
                    try :
                        l = monitor_params[sect][op].split(',')
                        monitor_params[sect][op] = np.array(l, dtype=np.int)
                        continue
                    except :
                        pass

    return monitor_params


def parse_cmdline_args_phasing():
    import argparse
    import os
    parser = argparse.ArgumentParser(prog = 'phase.py', description='phase a merged 3D diffraction volume')
    parser.add_argument('input', type=str, \
                        help="h5 file name of the input file")
    args = parser.parse_args()
    return args


def write_output_h5(path, diff, diff_ret, support, support_ret, \
        good_pix, solid_unit, solid_units_ret, emods, econs, efids, PRTF, PRTF_rav, PSD, PSD_I, B_rav):
    import os, h5py
    fnam = os.path.join(path, 'output.h5')
    if_exists_del(fnam)
    
    emods = np.array(emods)
    econs = np.array(econs)
    
    f = h5py.File(fnam, 'w')
    f.create_dataset('data', chunks = diff.shape, data = diff, compression='gzip')
    f.create_dataset('data retrieved', chunks = diff.shape, data = diff_ret, compression='gzip')
    f.create_dataset('sample support', chunks = support.shape, data = support.astype(np.int16), compression='gzip')
    f.create_dataset('sample support retrieved', chunks = support_ret.shape, data = support_ret, compression='gzip')
    f.create_dataset('good pixels', chunks = good_pix.shape, data = good_pix.astype(np.int16), compression='gzip')
    f.create_dataset('modulus error', chunks = emods.shape, data = emods, compression='gzip')
    f.create_dataset('convergence metric', chunks = econs.shape, data = np.array(econs), compression='gzip')
    if efids is not None :
        efids = np.array(efids)
        f.create_dataset('fidelity error', chunks = efids.shape, data = efids, compression='gzip')
    else :
        f.create_dataset('fidelity error', chunks = emods.shape, data = -np.ones_like(emods), compression='gzip')
    if solid_unit is not None:
        f.create_dataset('sample init', chunks = solid_unit.shape, data = solid_unit, compression='gzip')
    else:
        f.create_dataset('sample init', data= 'Randomly initiate')
    f.create_dataset('sample retrieved', chunks = (1,) + solid_units_ret.shape[1 :], data = solid_units_ret, compression='gzip')
    if PRTF is not None and PRTF_rav is not None :
        f.create_dataset('PRTF', chunks = PRTF.shape, data = PRTF, compression='gzip')
        f.create_dataset('PRTF radial average', chunks = PRTF_rav.shape, data = PRTF_rav, compression='gzip')
    if PSD is not None and PSD_I is not None :
        f.create_dataset('PSD', chunks = PSD.shape, data = PSD, compression='gzip')
        f.create_dataset('PSD data', chunks = PSD_I.shape, data = PSD_I, compression='gzip')
    if B_rav is not None :
        f.create_dataset('background radial average', chunks = B_rav.shape, data = B_rav, compression='gzip')

    # read the config file and dump it into the h5 file
    """
    g = open(config).readlines()
    h = ''
    for line in g:
        h += line
    f.create_dataset('config file', data = np.array(h))
    """
    f.close()
    return 


def read_output_h5(path):
    import os, h5py
    f = h5py.File(path, 'r')
    diff            = f['data'][()]
    diff_ret        = f['data retrieved'][()]
    support         = f['sample support'][()].astype(np.bool)
    support_ret     = f['sample support retrieved'][()]
    good_pix        = f['good pixels'][()].astype(np.bool)
    emods           = f['modulus error'][()]
    econs           = f['convergence metric'][()]
    efids           = f['fidelity error'][()]
    solid_unit      = f['sample init'][()]
    solid_units_ret = f['sample retrieved'][()]
    if 'PRTF' in f.keys():
        T               = f['PRTF'][()]
        T_rav           = f['PRTF radial average'][()]
    else :
        T = T_rav = None
    if 'PSD' in f.keys():
        PSD               = f['PSD'][()]
        PSD_rav           = f['PSD data'][()]
    else :
        PSD = PSD_rav = None
    if 'background radial average' in f.keys():
        B_rav           = f['background radial average'][()]
    else :
        B_rav = None
    #config_file    = f['config file'][()]

    f.close()
    
    return diff, diff_ret, support, support_ret, \
        good_pix, solid_unit, solid_units_ret, emods, econs, efids, T, T_rav, PSD, PSD_rav, B_rav


def write_input_h5(path, diff, support, good_pix, solid_known, config):
    import os, h5py
    fnam = os.path.join(path, 'input.h5')
    if_exists_del(fnam)
    
    f = h5py.File(fnam, 'w')
    f.create_dataset('data', data = diff)
    f.create_dataset('sample support', data = support.astype(np.int16))
    f.create_dataset('good pixels', data = good_pix.astype(np.int16))
    if solid_known is not None :
        f.create_dataset('sample', data = solid_known)
    # read the config file and dump it into the h5 file
    g = open(config).readlines()
    h = ''
    for line in g:
        h += line
    f.create_dataset('config file', data = h.encode())
    f.close()
    return 


def read_input_h5(fnam):
    import h5py
    
    f = h5py.File(fnam, 'r')
    diff     = f['data'][()]
    support  = f['sample support'][()].astype(np.bool)
    good_pix = f['good pixels'][()].astype(np.bool)
    
    if 'sample' in f.keys():
        solid_known = f['sample'][()]
    else :
        solid_known = None

    config_file = f['config file'][()]

    f.close()

    # read then pass the config file
    import configparser
    from io import StringIO
    config_file = StringIO(config_file.decode())

    config = configparser.ConfigParser()
    config.readfp(config_file)
    params = parse_parameters(config)
    return diff, support, good_pix, solid_known, params


def binary_out(array, fnam, endianness='little', appendType=True, appendDim=True):
    """Write a n-d array to a binary file."""
    arrayout = np.array(array)
    
    if appendDim == True :
        fnam_out = fnam + '_'
        for i in arrayout.shape[:-1] :
            fnam_out += str(i) + 'x' 
        fnam_out += str(arrayout.shape[-1]) + '_' + str(arrayout.dtype) + '.bin'
    else :
        fnam_out = fnam
    
    if sys.byteorder != endianness:
        arrayout.byteswap(True)
    
    arrayout.tofile(fnam_out)


def binary_in(fnam, ny = None, nx = None, dtype = None, endianness='little', dimFnam = True):
    """Read a n-d array from a binary file."""
    if dimFnam :
        # grab the dtype from the '_float64.bin' at the end
        tstr = fnam[:-4].split('_')[-1]
        if dtype is None :
            dtype = np.dtype(tstr)
        
        # get the dimensions from the 'asfasfs_89x12x123_' bit
        b    = fnam[:fnam.index(tstr)-1].split('_')[-1]
        dims = b.split('x')
        dims = np.array(dims, dtype=np.int)
        dims = tuple(dims)
        
        arrayout = np.fromfile(fnam, dtype=dtype).reshape( dims )
    
    else :
        arrayout = np.fromfile(fnam, dtype=dtype).reshape( (ny,nx) )
    
    if sys.byteorder != endianness:
        arrayout.byteswap(True)
    
    return arrayout


def if_exists_del(fnam):
    import os
    # check that the directory exists and is a directory
    output_dir = os.path.split( os.path.realpath(fnam) )[0]
    if os.path.exists(output_dir) == False :
        raise ValueError('specified path does not exist: ', output_dir)
    
    if os.path.isdir(output_dir) == False :
        raise ValueError('specified path is not a path you dummy: ', output_dir)
    
    # see if it exists and if so delete it 
    # (probably dangerous but otherwise this gets really anoying for debuging)
    if os.path.exists(fnam):
        print('\n', fnam ,'file already exists, deleting the old one and making a new one')
        os.remove(fnam)
