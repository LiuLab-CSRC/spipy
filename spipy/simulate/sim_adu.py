import numpy as np
import os
import json
from ..analyse import rotate
from scipy.linalg import get_blas_funcs
import numexpr as ne
from mpi4py import MPI

comm = MPI.COMM_WORLD
m_rank = comm.Get_rank()
m_size = comm.Get_size()

def help(module):
    if module=="go_magic":
        print("This function generates spi patterns by simulating atom reflection using single/multiple jobs")
        print("    -> Input: method ('atomic' or 'fft', which means atomic diffraction and fourier transform)")
        print("              save_dir (path to save simulation result, a directory path)")
        print("              pdb_file (path of your input pdb file)")
        print("              param (simulation parameters, the same with 'spipy.simulate.sim' module)")
        print("     #option: euler_mode (how to genreate euler angles, str, chosen from 'random', 'helix' and 'predefined', default='random')")
        print("     #option: euler_order (rotation order, such as 'zxz' or 'zyz'..., default='zxz')")
        print("     #option: euler_range (the range of euler angles to rotate object, numpy.array([[alpha_min,alpha_max],[beta_min,beta_max],[gamma_min,gamma_max]]), shape=(3,2)), default=None")
        print("     #option: predefined (if euler_mode is 'predefined', then you have to specify all euler angles used for object rotation, shape=(Ne,3), default=None)")
        print("    -> NO RETURN")
    else:
        raise ValueError("No module names "+str(module))


class simulation():

    sim_methods = ["atomic", "fft"]
    config_param = {'detd' : 200.0, 'lambda' : 2.5, \
                    'detsize' : 128, 'pixsize' : 0.3, \
                    'stoprad' : 0, 'polarization' : 'x', \
                    'num_data' : 100, 'fluence' : 1.5e14, \
                    'photons' : False, 'absorption' : True, \
                    'phy.scatter_factor' : True, 'phy.ram_first' : True, \
                    'phy.projection' : True }
    # template variables
    pdb_file = None  # pdb file name
    euler = None  # [intrisinc] alpha 0,2pi ; beta 0,pi ; gamma 0,2pi ; shape=(Ne,3)
    order = None  # rotation order 'zxz','zyz',...
    atoms = {'coordinate':None, \
            'index':None, \
            'obs_electron':None} # (angstrom) coordinate.shape=(Na,3), obs_electron : observed electron number
    sample_size = None  # (nanometer)
    oversampl = None
    inten_correction = None # np.array, shape=(detsize,detsize)

    def __init__(self, method="atomic"):
        self.set_method(method)

    def __show_info(self, kwargs):
        det_l = self.config_param['detsize']
        fmt = "%-25s: %-10s"
        print("Simulation Information :")
        print(fmt % ("PDB file", self.pdb_file))
        print(fmt % ("Method", simulation.sim_methods[self.__method]))
        print(fmt % ("Detector Size", str((det_l,det_l))))
        print(fmt % ("Detector Distance (mm)", str(self.config_param['detd'])))
        print(fmt % ("Pixel Size (mm)", str(self.config_param['pixsize'])))
        print(fmt % ("Wave Length (A)", str(self.config_param['lambda'])))
        print(fmt % ("Save Photons", str(self.config_param['photons'])))
        print(fmt % ("Polarization", self.config_param['polarization']))
        print(fmt % ("Absorption", self.config_param['absorption']))
        print(fmt % ("Laser Fluence", "%.2e" % self.config_param['fluence']))
        for k, v in kwargs.items():
            print(fmt % (k, str(v)))
        print("Done.\n")

    def set_method(self, method):
        if not method in simulation.sim_methods:
            raise ValueError("Unknown simulation method. Chosen from %s" % str(simulation.sim_methods))
        self.__method = simulation.sim_methods.index(method)

    def configure(self, pdb_file, param):
        from ..analyse import q
        from ..image import io
        # read parameters
        for key, value in param.items():
            if key in self.config_param.keys():
                if type(value)==type(self.config_param[key]):
                    self.config_param[key] = value
                else:
                    raise ValueError("The data type of '"+key+"' in param is wrong")
            else:
                raise ValueError('param has unknown items')
        self.k0 = np.array([0,0,1])
        self.inten_correction = self.cal_inten_correction()  # shape=(detsize,detsize)
        # read pdb
        if os.path.exists(pdb_file):
            if self.config_param['absorption']:
                # columns = (index,x,y,z,obs_electron,mass)
                atom_info = io._readpdb(pdb_file, self.config_param['lambda'])
                self.atoms['obs_electron'] = atom_info[:,4]
            else:
                # columns = (index,x,y,z,index,mass)
                atom_info = io._readpdb(pdb_file, None)
            self.atoms['coordinate'] = atom_info[:,1:4]
            self.atoms['index'] = np.round(atom_info[:,0]).astype(int)
            self.pdb_file = pdb_file
        else:
            raise ValueError('pdb file path is invalid')
        # put molecular to the coordinate origin
        center = self.atoms['coordinate'].mean(axis=0)
        self.atoms['coordinate'] -= center
        # calculate oversampling rate
        self.sample_size = np.linalg.norm(self.atoms['coordinate'], axis=1).max()*2/10.  # in nanometer
        self.oversampl = q.oversamp_rate(self.sample_size, self.config_param['detd'], \
                self.config_param['lambda'], self.config_param['pixsize'])

    def cal_inten_correction(self):
        # returned correction should be multiplied to diffraction intensities
        from ..image.preprocess import cal_correction_factor
        correction = cal_correction_factor([self.config_param['detsize'], self.config_param['detsize']], \
                                            self.config_param['polarization'], \
                                            self.config_param['detd'], \
                                            self.config_param['pixsize'])
        return correction

    def generate_euler(self, mode='random', order='zxz', arange=None, predefined=None): 
        if arange is None:
            if mode=='random':
                pat_num = self.config_param['num_data']
                alpha = np.random.rand(pat_num) * 2 * np.pi
                beta = np.arccos( ( np.random.rand(pat_num) - 0.5 ) * 2 )
                gamma = np.random.rand(pat_num) * 2 * np.pi
                self.euler = np.vstack([alpha,beta,gamma]).T
            elif mode=='helix':
                pat_num = self.config_param['num_data']
                alpha = np.linspace(0,np.pi*2,pat_num)
                beta = np.arccos(np.linspace(-1,1,pat_num))[::-1]
                gamma = np.linspace(0,np.pi*2,pat_num)
                self.euler = np.vstack([alpha,beta,gamma]).T
            elif mode=='predefined':
                if predefined is None:
                    raise RuntimeError("I need euler angles you defined in mode 'predefined'")
                elif len(np.array(predefined).shape)!=2 or np.array(predefined).shape[1]!=3:
                    raise RuntimeError("I can't recognize your input euler angles ...")
                else:
                    self.euler = np.array(predefined)
                    self.config_param['num_data'] = self.euler.shape[0]
            else:
                raise RuntimeError("I can't recognize your mode")
        elif arange.shape==(3,2):
            if mode=='random':
                pat_num = self.config_param['num_data']
                alpha = np.random.rand(pat_num) * (arange[0].max()-arange[0].min()) + arange[0].min()
                beta = np.arccos((np.random.rand(pat_num)-0.5)*2)/np.pi * (arange[1].max()-arange[1].min()) + arange[1].min()
                gamma = np.random.rand(pat_num) * (arange[2].max()-arange[2].min()) + arange[2].min()
                self.euler = np.vstack([alpha,beta,gamma]).T
            elif mode=='helix':
                pat_num = self.config_param['num_data']
                alpha = np.linspace(arange[0].min(),arange[0].max(),pat_num+2)[1:-1]
                beta = np.arccos(np.linspace(-1,1,pat_num+2))[::-1]/np.pi*(arange[1].max()-arange[1].min()) + arange[1].min()
                beta = beta[1:-1]
                gamma = np.linspace(arange[2].min(),arange[2].max(),pat_num+2)[1:-1]
                self.euler = np.vstack([alpha,beta,gamma]).T
            elif mode=='predefined':
                if predefined is None:
                    raise RuntimeError("I need euler angles you defined in mode 'predefined'")
                elif len(np.array(predefined).shape)!=2 or np.array(predefined).shape[1]!=3:
                    raise RuntimeError("I can't recognize your predefined euler angles ...")
                else:
                    self.euler = np.array(predefined)
                    self.config_param['num_data'] = self.euler.shape[0]
            else:
                raise RuntimeError("I can't recognize your mode")
        else:
            raise RuntimeError('Your input euler angle range is invalid')
        if len(self.euler) == self.config_param['num_data']:
            self.order = order
        else:
            raise RuntimeError("Here occurs some error in configuration ... error-code:s2_geneul. Please report this bug")

    def generate_box(self, den_map, box_size=None):
        # generate fittable box for density map beore doing FFT
        if len(den_map.shape)!=3 or (box_size is not None and len(box_size)!=3):
            raise RuntimeError('Your input density map or box_size is invalid')
        center = np.array(den_map.shape)//2
        if box_size is None:
            box_size = (np.array(den_map.shape)*self.oversampl).astype(int)
        box = np.zeros(box_size)
        box_cen = np.array(box.shape)//2
        box[box_cen[0]-center[0]:box_cen[0]+den_map.shape[0]-center[0],\
            box_cen[1]-center[1]:box_cen[1]+den_map.shape[1]-center[1],\
            box_cen[2]-center[2]:box_cen[2]+den_map.shape[2]-center[2]] = den_map
        return box

    def get_scatt(self, atom_index, pix_r):
        '''
        return a lookup table, shape=(max(atom_index)+1,max(pix_r)+1)
        [NOTICE!] for those atom index or q that are not used in simulation, the value of corresponding cell
        of the return table is ZERO ! AND, the atom index start from 1 (which means the first row of the
        returned table is always zero)!
        '''
        def gaussian(a, b, c, q):
            a = a.reshape((len(a),1))
            b = b.reshape((len(b),1))
            c = c.reshape((len(c),1))
            q = q.reshape((len(q),))
            return a*np.exp(-b*q**2)+c

        det_lambda = self.config_param['lambda']
        det_d = self.config_param['detd']
        det_ps = self.config_param['pixsize']

        abs_index = np.sort(list(set(atom_index))).astype(int)
        abs_r = np.sort(list(set(pix_r))).astype(int)
        q = np.sin(np.arctan(abs_r*det_ps/det_d)/2.0)/det_lambda
        scatter_file = os.path.join(os.path.dirname(__file__), 'aux/scattering.json')
        with open(scatter_file, 'r') as fp:
            scatterf = json.load(fp)
        abc = np.zeros((len(abs_index), 9))
        for ind,k in enumerate(abs_index):
            abc[ind] = scatterf[str(k)]
        gau_1 = gaussian(abc[:,0],abc[:,1],abc[:,8],q)
        gau_2 = gaussian(abc[:,2],abc[:,3],abc[:,8],q)
        gau_3 = gaussian(abc[:,4],abc[:,5],abc[:,8],q)
        gau_4 = gaussian(abc[:,6],abc[:,7],abc[:,8],q)
        fq = gau_1 + gau_2 + gau_3 + gau_4
        Fq = np.zeros((np.max(abs_index)+1,np.max(abs_r)+1))
        Fq[abs_index.reshape((len(abs_index),1)),abs_r] = fq
        return Fq

    def rotate_mol(self, euler_angle, coor=None):
        '''
            coor : the coordinates that you want to rotate, in default atoms' coordinates are used
                   coor.shape should be (N,3)
        '''
        if self.order is None or self.atoms['index'] is None:
            raise RuntimeError('Please configure and generate euler first!')
        if coor is not None:
            old_coor = coor
        else:
            old_coor = self.atoms['coordinate']   # shape=(N, 3)
        rotate_matrix = rotate.eul2rotm(euler_angle, self.order)
        new_coor = rotate_matrix.dot(old_coor.T).T
        return new_coor

    def generate_pattern_fft(self, euler_angles, verbose):
        from ..analyse import q
        from ..image import io
        from ..merge import tools
        import time

        if self.order is None or self.atoms['index'] is None:
            raise RuntimeError('Please configure and generate euler first!')

        time0 = time.time()
        # get parameters
        det_l = self.config_param['detsize']
        det_d = np.float32(self.config_param['detd'])
        det_ps = np.float32(self.config_param['pixsize'])
        det_lambda = np.float32(self.config_param['lambda'])
        to_photons = self.config_param['photons']
        # get pixel coordinates in k-space of our detector with rotate angle=0
        q_coor, q_max, _, q_len = q.ewald_mapping(det_d, det_lambda, det_ps, [det_l, det_l], center=None)
        q_coor = q_coor.reshape([det_l*det_l,3])

        # print information
        total_num_data = comm.gather(self.config_param['num_data'], root=0)
        if m_rank == 0:
            infomation = {}
            if verbose:
                infomation = {"Diffraction Volum" : [q_len]*3, \
                            "Corner Resolution (A)" : "%.2f" % (10.0/q_max), \
                            "Oversampling Rate" : "%.2f" % self.oversampl, \
                            "Num Patterns" : np.sum(total_num_data, dtype=int)}
            self.__show_info(infomation)
            print("Simulation starts ...")

        intensity = None
        if m_rank == 0:
            # transfer pdb to density
            density = io.pdb2density(self.pdb_file, 5.0/q_max, det_lambda)
            density = self.generate_box(density, [q_len]*3)
            # doing fft and slicing
            intensity = np.abs(np.fft.fftshift(np.fft.fftn(density))) ** 2

        intensity = comm.bcast(intensity, root=0)
        patterns = tools.get_slice(model=intensity, rotations=euler_angles, det_size=[det_l, det_l], \
                            mask=None, slice_coor_ori=q_coor.T, euler_order=self.order)
        patterns *= self.inten_correction   # patterns.shape=(num,det_l,det_l)
        # evaluate adu and add noise
        photons = float(self.config_param['fluence'] / 1e10)
        adu = photons * 12398.419 / det_lambda / 1e2
        if to_photons:
            patterns = np.random.poisson( patterns * photons / np.sum(patterns, axis=(1,2)).reshape((len(patterns),1,1)) )
        else:
            patterns = patterns * adu / np.sum(patterns, axis=(1,2)).reshape((len(patterns),1,1))

        if verbose:
            print("Rank %d finished simulation, time used : %.2f seconds" % (m_rank, time.time()-time0))
        else:
            print("Rank %d finished." % m_rank)
        return patterns

    def generate_pattern_atomic(self, euler_angles, patt, projections, verbose):
        import psutil
        import time
        import gc
        from ..analyse import q

        if self.order is None or self.atoms['index'] is None:
            raise RuntimeError('Please configure and generate euler first!')
        # get parameters
        det_l = self.config_param['detsize']
        det_d = np.float32(self.config_param['detd'])
        det_ps = np.float32(self.config_param['pixsize'])
        det_lambda = np.float32(self.config_param['lambda'])
        to_photons = self.config_param['photons']
        sampling_len = 0.5/(q.cal_q(det_d,det_lambda,det_l/2.0,det_ps).max())   # in nm, considering Nquist sampling requirement
        ati = self.atoms['index']
        ati = ati.reshape((len(ati),1))
        if self.config_param['absorption']:
            obs_ati = self.atoms['obs_electron']
            obs_ati = obs_ati.reshape((len(obs_ati),1))
        else:
            obs_ati = ati
        # calculate detector information
        det_cen = (det_l-1)/2.0
        detx, dety = np.mgrid[0:det_l,0:det_l]        # here in pixel
        detx = detx.flatten()
        dety = dety.flatten()
        if self.config_param['phy.scatter_factor'] is True:
            pix_r = np.sqrt((detx-det_cen)**2 + (dety-det_cen)**2).astype(int)
            scatt = self.get_scatt(self.atoms['index'], pix_r)  # scattering factor, shape=(max(atom_index)+1,max(pix_r)+1)
            if self.config_param['absorption']:
                scaling_obs = obs_ati / ati
        else:
            scatt = obs_ati     # shape=(Nr,1)
        screen_xy = np.array([detx-det_cen, dety-det_cen]).T * det_ps   # here in mm
        screen_xyz = np.hstack([screen_xy, np.zeros((det_l**2,1))+det_d])        # here in mm
        k_prime = screen_xyz/np.linalg.norm(screen_xyz,axis=1).reshape((len(screen_xyz),1)) # here no unit
        dk = (k_prime - self.k0)/det_lambda   # dk.shape=(Nk,3) the value of dk of every pixel on detector, angstrom^(-1)

        # print information
        total_num_data = comm.gather(self.config_param['num_data'], root=0)
        if m_rank == 0:
            resolution = 10.0/(q.cal_q(det_d,det_lambda,det_l/np.sqrt(2),det_ps).max())
            infomation = {}
            if verbose:
                infomation = {"Corner Resolution (A)" : "%.2f" % resolution, \
                            "Oversampling Rate" : "%.2f" % self.oversampl, \
                            "Num Patterns" : np.sum(total_num_data, dtype=int)}
            self.__show_info(infomation)
            print("Simulation starts ...")
        
        # now start loop
        for ind,euler_angle in enumerate(euler_angles):

            print('\nRank %d is generating %dth patterns' %(m_rank, ind+1))
            time0 = time.time()
            dr = self.rotate_mol(euler_angle) # dr.shape=(Nr,3) the position vector of atoms, in angstrom
            if projections is not None:
                # toy projection of the molecule onto the xOy plane 
                dr_ = dr / (sampling_len * 10)
                projection = np.zeros((det_l,det_l))
                dr_projection = (np.round(dr_[:,0:2] + (det_l-1)/2.0)).astype(int)
                for ii, xy in enumerate(dr_projection):
                    projection[xy[0],xy[1]] += ati[ii]
                projections[ind] = projection

            pat = np.zeros(len(dk),dtype=np.complex64)
            if self.config_param['phy.ram_first']:
                # forced to use cblas lib to accelerate
                gemm = get_blas_funcs("gemm",[dr,dk[0].reshape((3,1))])
                for ii,dkk in enumerate(dk):
                    temp = np.complex128(1j)*-1*2*np.pi*gemm(1,dr,dkk.reshape((3,1)))    # temp.shape=(Nr,1)
                    temp = ne.evaluate('exp(temp)')
                    if self.config_param['phy.scatter_factor'] is True:
                        temp *= scatt[ati,pix_r[ii]].reshape(temp.shape)       # (Nr,1)
                        if self.config_param['absorption']:
                            temp *= scaling_obs                          # scaling_obs.shape=(Nr,1)
                    else:
                        temp *= scatt
                    pat[ii] = np.sum(temp)
                if verbose: print(u"RAM (MB): ",psutil.Process(os.getpid()).memory_info().rss/1024.0**2)
            else:
                # forced to use cblas lib to accelerate
                gemm = get_blas_funcs("gemm",[dr.astype(np.float32),dk.T.astype(np.float32)])
                pat = np.complex128(1j)*-1*2*np.pi*gemm(1,dr,dk.T)    # matrix shape=(Nr,Nk)
                pat = ne.evaluate('exp(pat)')

                if self.config_param['phy.scatter_factor'] is True:
                    pat *= scatt[ati, pix_r]
                    if self.config_param['absorption']:
                            pat = pat * scaling_obs 
                else:
                    pat *= scatt
                if verbose: print(u"RAM (MB): ",psutil.Process(os.getpid()).memory_info().rss/1024.0**2)
                pat = np.sum(pat,axis=0)

            pat = pat.reshape((det_l, det_l))
            pat = np.abs(pat)**2 * self.inten_correction
            # evaluate adu and add noise
            photons = float(self.config_param['fluence'] / 1e10)
            adu = photons * 12398.419 / det_lambda / 1e2
            if to_photons:
                patt[ind] = np.random.poisson( pat * photons/np.sum(pat) ).astype(float)
            else:
                patt[ind] = pat * adu/np.sum(pat)

            del pat
            gc.collect()
            if verbose: print('Done. Time (s): '+str(time.time()-time0))

    def run_simulation(self, verbose=True):                                                       
        if self.order is None or self.euler is None or self.oversampl is None:
            raise RuntimeError('Please configure and generate euler angles first!')
        from spipy.image import radp

        num_pat = self.config_param['num_data']
        det_l = self.config_param['detsize']
        bstopR = self.config_param['stoprad']
        patterns = None

        if self.__method == 0:
            # atomic
            if self.config_param['phy.projection']:
                projections = np.zeros((num_pat, det_l, det_l))
            else:
                projections = None
            # generate patterns
            patterns = np.zeros((num_pat, det_l, det_l))
            self.generate_pattern_atomic(self.euler, patterns, projections, verbose)
        else:
            # fft
            projections = None
            # generate patterns
            patterns = self.generate_pattern_fft(self.euler, verbose)

        if bstopR is not None and bstopR > 0:
            beam_stop = radp.circle(2, bstopR) + np.array(patterns.shape[1:])//2
            patterns[:,beam_stop[:,0],beam_stop[:,1]] = 0

        output = {'oversampling_rate' : self.oversampl, \
                    'euler_angles' : self.euler, \
                    'rotation_order' : self.order, \
                    'simu_parameters' : self.config_param, \
                    'patterns' : patterns, \
                    'projections':projections}
        return output


def __single_process(method, pdb_file, param, euler_mode='random', euler_order='zxz', euler_range=None, predefined=None, save_dir=None, verbose=True):
    import h5py
    import time

    if save_dir is not None and not os.path.isdir(save_dir):  
        raise ValueError('Your save directory is invalid')

    sim = simulation(method=method)
    sim.configure(pdb_file, param)
    sim.generate_euler(euler_mode, euler_order, euler_range, predefined)
    dataset = sim.run_simulation(verbose)

    if save_dir is not None:
        # save h5
        time_seed = time.ctime()
        time_seed = time_seed.replace(' ','_')
        time_seed = time_seed.replace(':','_')
        savef = h5py.File(os.path.join(save_dir, 'spipy_adu_simulation_'+time_seed+'.h5'), 'w')
        savef.create_dataset('oversampling_rate', data=dataset['oversampling_rate'])
        savef.create_dataset('rotation_order', data=dataset['rotation_order'])
        savef.create_dataset('patterns', data=dataset['patterns'], chunks=True, compression="gzip")
        if dataset['projections'] is not None:
            savef.create_dataset('projections', data=dataset['projections'], chunks=True, compression="gzip")  
        savef.create_dataset('euler_angles', data=dataset['euler_angles'], chunks=True, compression="gzip")
        grp = savef.create_group('simu_parameters')
        for k, v in dataset['simu_parameters'].items():
            grp.create_dataset(k, data=v)
        savef.close()
        return
    else:
        return dataset


def go_magic(method, save_dir, pdb_file, param, euler_mode='random', euler_order='zxz', euler_range=None, predefined=None, verbose=True):
    import h5py
    import time
    import sys

    if m_rank == 0:
        if not os.path.isdir(save_dir):
            raise ValueError('Your save directory is invalid')

    if m_size == 1:
        __single_process(method, pdb_file, param, euler_mode, euler_order, euler_range, predefined, save_dir, verbose)
        sys.exit(0)

    if euler_mode=='predefined':
        num_pat = len(predefined)
    else:
        num_pat = param['num_data']

    pool_bin = np.linspace(0, num_pat, m_size+1, dtype=int)

    # split num_pat
    if m_rank == 0:

        # split euler
        if euler_mode=='random' or euler_mode=='helix':
            if euler_range is not None:
                alpha_bin = np.linspace(euler_range[0,0], euler_range[0,1], m_size+1)
                beta_bin = np.linspace(euler_range[1,0], euler_range[1,1], m_size+1)
                gamma_bin = np.linspace(euler_range[2,0], euler_range[2,1], m_size+1)
            else:
                alpha_bin = np.linspace(0, np.pi*2, m_size+1)
                beta_bin = np.linspace(0, np.pi, m_size+1)
                gamma_bin = np.linspace(0, np.pi*2, m_size+1)

        # generate h5 name, and write dataset structure
        time_seed = time.ctime()
        time_seed = time_seed.replace(' ','_')
        time_seed = time_seed.replace(':','_')
        savefilename = os.path.join(save_dir, 'spipy_adu_simulation_'+time_seed+'.h5')

        savef = h5py.File(savefilename, 'w')
        grp = savef.create_group('simu_parameters')
        for k, v in param.items():
            grp.create_dataset(k, data=v)
        savef.create_dataset('patterns', (num_pat, \
            param['detsize'], param['detsize']), \
            dtype=float, chunks=True, compression="gzip")
        savef.create_dataset('euler_angles', (num_pat, 3), \
            dtype=float, chunks=True, compression="gzip")
        if method == simulation.sim_methods[0] and param["phy.projection"]:
            savef.create_dataset('projections', (num_pat, \
                param['detsize'], param['detsize']), \
                dtype=float, chunks=True, compression="gzip")
        savef.close()

        # mpi
        for ind, i in enumerate(pool_bin[:-1]):
            param_copy = param.copy()
            param_copy['num_data'] = int(pool_bin[ind+1]-i)
            if euler_mode=='random' or euler_mode=='helix':
                euler_range_copy = np.array([alpha_bin[ind:ind+2], beta_bin[ind:ind+2], gamma_bin[ind:ind+2]])
                euler_defined_copy = None
            elif euler_mode=='predefined':
                euler_defined_copy = predefined[i:pool_bin[ind+1]]
                euler_range_copy = None
            comm.send([param_copy, euler_range_copy, euler_defined_copy, savefilename], dest=ind)

    # recv
    param_copy, euler_range_copy, euler_defined_copy, savefilename = comm.recv(source=0)

    # run single_process
    solution = __single_process(method, pdb_file, param_copy, euler_mode, euler_order, euler_range_copy, euler_defined_copy, None, verbose)

    # write to h5
    savef = None
    while savef is None:
        try:
            savef = h5py.File(savefilename, 'a')
        except:
            savef = None
    
    savef['patterns'][pool_bin[m_rank]:pool_bin[m_rank+1]] = solution['patterns']
    if solution['projections'] is not None:
        savef['projections'][pool_bin[m_rank]:pool_bin[m_rank+1]] = solution['projections']
    savef['euler_angles'][pool_bin[m_rank]:pool_bin[m_rank+1]] = solution['euler_angles']
    savef.close()

    comm.Barrier()

    # write additional infomation to output h5
    if m_rank == 0:
        oversampling_rate = solution['oversampling_rate']
        savef = h5py.File(savefilename, 'a')
        savef.create_dataset('oversampling_rate', data=oversampling_rate)
        savef.create_dataset('rotation_order', data=euler_order)
        savef.close()

    MPI.Finalize()