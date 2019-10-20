from __future__ import print_function, division, absolute_import

import numpy as np
from spipy.simulate import sim_adu

if __name__ == '__main__':
    pdb = "./ico.pdb"
    config_param = {'parameters|detd' : 600.0, 'parameters|lambda' : 5.0, \
        'parameters|detsize' : 128, 'parameters|pixsize' : 1.2, \
        'parameters|stoprad' : 0, 'parameters|polarization' : 'x', \
        'make_data|num_data' : 6, 'make_data|fluence' : 1.0e15, \
        'make_data|scatter_factor' : True, 'make_data|ram_first' : True, \
        'make_data|poisson' : True}
    euler_range = np.array([[0, np.pi / 2.0], [0, np.pi / 2.0],
                            [0, np.pi / 2.0]])
    euler = [[0,0,1.57],[1.57,0,0],[0.754,0,0],[0,0.754,0],[1,1,1]]
    '''
    sim_adu.single_process(pdb_file=pdb, param=config_param, euler_mode='random', \
        euler_order='zxz', euler_range=euler_range, predefined=None, save_dir='./', verbose=True)
    '''
    sim_adu.multi_process(save_dir='./', pdb_file=pdb, param=config_param, euler_mode='predefined', \
        euler_order='zxz', euler_range=euler_range, predefined=euler, verbose=False)
    
