import numpy as np
from spipy.simulate import sim_adu

if __name__ == '__main__':
    pdb = "./ico.pdb"
    methods = ["fft", "atomic"]
    config_param = {'detd' : 600.0, 'lambda' : 5.0, \
        'detsize' : [128,130], 'pixsize' : 1.2, \
        'stoprad' : 3, 'polarization' : 'x', \
        'num_data' : 5, 'fluence' : 1.0e11, 'absorption' : True, \
        'adu_per_eV' : 0.001, 'detcenter' : [63.2,63.7], \
        'phy.scatter_factor' : True, 'phy.ram_first' : True, \
        'photons' : False, 'phy.projection' : True, \
        'phy.b_factor' : 20.0}
    euler_range = np.array([[0, np.pi], [0, np.pi*2],
                            [0, np.pi*2]])
    euler = [[0,0,1.57],[1.57,0,0],[0.754,0,0],[0,0.754,0],[1,1,1]]

    # start
    sim_adu.go_magic(method=methods[1], save_dir='./', pdb_file=pdb, \
                    param=config_param, euler_mode='random', euler_order='zxz', \
                    euler_range=euler_range, predefined=euler, verbose=True)