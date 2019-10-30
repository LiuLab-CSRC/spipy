#!/usr/bin/env python

import time
import sys
import os
import numpy as np
import configparser
import scipy

from .utils import io_utils, zero_pad, circle

def make_input(args_config):

    config = configparser.ConfigParser()
    config.read(args_config)
    
    params = io_utils.parse_parameters(config)

    # read the *.bin file
    dtype = np.dtype(params['input']['dtype'])
    shape = params['input']['shape']
    print('\n Loading:', params['input']['fnam'], 'as', dtype, 'with shape', shape)
    diff = np.fromfile(params['input']['fnam'], dtype=dtype).reshape(shape)

    # padd to the nearest power of 2
    if params['input']['padd_to_pow2'] is True : 
        print('\n padding...')
        diff = zero_pad.zero_pad_to_nearest_pow2(diff)

    # fft shift
    print('\n fft shifting the zero frequency to the centre of the array:')
    diff = np.fft.ifftshift(diff)

    # padd to the nearest power of 2
    if params['input']['padd_to_pow2'] is True : 
        print('\n padding...')
        diff = zero_pad.zero_pad_to_nearest_pow2(diff)
    elif params['input']['padd_to_pow2'] is not None and params['input']['padd_to_pow2'] is not False:
        diff = zero_pad.zero_pad_to_nearest_pow2(diff, tuple(params['input']['padd_to_pow2']))

    # beamstop
    if params['input']['inner_mask'] is not None :
        print('\n masking the central',params['input']['inner_mask'],'pixels')
        beamstop = circle.make_beamstop(diff.shape, params['input']['inner_mask'])
    else :
        beamstop = np.ones_like(diff, dtype=np.bool)
    # user defined beamstop
    if params['input']['user_mask'] is not None:
        print("\n Applying user defined mask.")
        ubeamstop = np.load(params['input']['user_mask']).astype(bool)
        if params['input']['padd_to_pow2'] is True : 
            ubeamstop = zero_pad.zero_pad_to_nearest_pow2(ubeamstop)
        ubeamstop = ~ubeamstop
        ubeamstop = np.fft.ifftshift(ubeamstop)
        beamstop *= ubeamstop

    # set the outer pixels to zero
    if params['input']['outer_mask'] is not None :
        print('\n setting the outer',params['input']['outer_mask'],'pixels to zero')
        diff *= ~circle.make_beamstop(diff.shape, params['input']['outer_mask'])

    # pixels between outer_mask-->outer_outer_mask are allowed to float
    if params['input']['outer_outer_mask'] is not None :
        print('\n allowing pixels between',params['input']['outer_mask'],\
                'and',params['input']['outer_outer_mask'],'to float')
        edges  = circle.make_beamstop(diff.shape, params['input']['outer_outer_mask'])
        edges += ~circle.make_beamstop(diff.shape, params['input']['outer_mask'])
    else :
        edges = np.ones_like(diff, dtype=np.bool)

    if params['input']['mask_edges'] is not None :
        if params['input']['mask_edges'] is True :
            beamstop *= edges

    # mask negative values and nans
    #beamstop[np.isnan(diff)] = False
    diff = np.nan_to_num(diff)
    diff[diff < 0.]      = 0.
    
    if params['input']['subtract_percentile'] is not None :
        print('\n subtracting the',params['input']['subtract_percentile'],\
                '\'th percentile from the diffraction intensities')
        p = np.percentile(diff[(beamstop > 0) * (diff > 0)], params['input']['subtract_percentile'])
        diff -= p
        diff[diff < 0] = 0

    if params['input']['spherical_support'] is not None :
        support = ~circle.make_beamstop(diff.shape, params['input']['spherical_support'])
    else :
        support = np.ones_like(diff, dtype=np.bool)

    # generate solid known
    if params['input']['init_model'] is not None:
        file_name = params['input']['init_model']
        if os.path.splitext(file_name)[1]=='npy':
            solid_known = np.load(file_name)
        elif os.path.splitext(file_name)[1]=='bin' or os.path.splitext(file_name)[1]=='':
            solid_known = np.fromfile(file_name, dtype=dtype).reshape(shape)
        elif os.path.splitext(file_name)[1]=='mat':
            solid_known = scipy.io.loadmat(file_name).values()[0]
        else:
            raise RuntimeError('Cannot open your input initial model file')
        if len(solid_known.shape)!=3:
            raise RuntimeError('Inital model should be a 3-dimension matrix')
    else:
        solid_known = None

    # write to file
    print('writing to file...', params['output']['path'])
    io_utils.write_input_h5(params['output']['path'], diff, support, \
            beamstop, solid_known, args_config)

'''
if __name__ == "__main__":
    args = io_utils.parse_cmdline_args()
    
    make_input(args.config)
'''
