import sys
import os
import argparse
import yaml
import h5py
from spipy.merge import emc

if __name__ == '__main__':

    # parse cmd arguments
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description = "Orientation recovery and 3D merging.")
    parser.add_argument("-i", "--inpath", type=str, help="Input HDF5 file to store patterns and mask.", required=True)
    parser.add_argument("-o", "--outpath", type=str, help="Output folder path.", required=True)
    parser.add_argument("-t", "--topol", type=str, help="Detector topology file.", required=True)
    parser.add_argument("--data_h5loc", type=str, help="The dataset location in HDF5 file.", required=True)
    parser.add_argument("--mask_h5loc", type=str, default="", help="The mask location in HDF5 file (if there is), default=''. Note in 'mask' array, 0 means unmasked pixel, 1 means masked pixel in orientation recovery calculation, 2 means masked pixel in both orientation recovery and merging.")
    parser.add_argument("--mask_file", type=str, default="", help="Mask file, if this is given, '--mask_h5loc' will be ignored, support '.npy' and binary data file, default=''.")
    parser.add_argument("--proj_name", type=str, default="", help="Give a specific name to this project, default name is like 'emc_xx'.")
    parser.add_argument("--num_div", type=int, default=10, help="Parameter for generating quaternions, M_rot=10(n+5n^3), default=10.")
    parser.add_argument("--beta", type=float, default=0.001, help="The EMC beta value, default=0.001.")
    parser.add_argument("--beta_schedule", type=str, default="1.414,10", help="How beta changes with iteration, 'k,i' means for every i iterations beta is multiplied by k, default='1.414,10'.")
    parser.add_argument("--polarization", type=str, choices=['x','y','none'], default='none', help="Laser polarization direction, 'x' or 'y' or 'none', default='none'.")
    parser.add_argument("--stoprad", type=float, default=0, help="Radius of a circle area at the center of pattern which will not be used in orientation recovery, but will be merged to final scattering volume. Default=0.")
    parser.add_argument("--ewald_rad", type=float, default=0, help="Radius of curvature of the Ewald sphere in voxels, default=detd/pixsize.")
    parser.add_argument("--selection", type=str, default='None', choices=['even_only', 'odd_only', 'None'], help="'even' / 'odd' means only patterns whose index is even / odd will be used. 'None' means all patterns will be used. Default is None.")
    parser.add_argument("--no_scaling", action="store_true", default=False, help="Set to skip scaling patterns.")
    parser.add_argument("--sym_icosahedral", action="store_true", default=False, help="Set to apply icosahedral symmetry to data.")
    args = parser.parse_args()

    # get parameters
    assert os.path.isfile(args.inpath), "Input HDF5 file is invalid !"
    input_h5 = args.inpath
    data_loc = args.data_h5loc
    assert os.path.isdir(args.outpath), "Output folder is invaid !"
    output_folder = os.path.abspath(args.outpath)
    if len(args.proj_name) > 0:
        proj_name = args.proj_name
    else:
        proj_name = None

    with open(args.topol,'r') as fp:
        topol = yaml.load(fp, Loader=yaml.FullLoader)
    config = {
        'parameters|detd' : topol['detd'], 'parameters|lambda' : topol['wavelength'], \
        'parameters|detsize' : '%d %d' % tuple(topol['detsize']), 'parameters|stoprad' : args.stoprad, \
        'parameters|pixsize' : topol['pixsize'], 'parameters|polarization' : args.polarization, \
        'emc|num_div' : args.num_div, 'emc|need_scaling' : int(not args.no_scaling), \
        'emc|beta' : args.beta, 'emc|beta_schedule' : '%s %s' % tuple(args.beta_schedule.split(',')), \
        'emc|sym_icosahedral' : int(args.sym_icosahedral), 'emc|selection' : args.selection
    }

    if topol['center'] is not None:
        config['make_detector|center'] = '%.1f %.1f' % tuple(topol['center'])
    if args.ewald_rad > 0:
        config['parameters|ewald_rad'] = args.ewald_rad
    if len(args.mask_file) > 0:
        assert os.path.isfile(args.mask_file), "Input mask file is invalid !"
        config['make_detector|in_mask_file'] = os.path.abspath(args.mask_file)
    else:
        try:
            with h5py.File(input_h5, 'r') as fp:
                mask = fp[args.mask_h5loc][()]
        except:
            raise ValueError("Check -i and --mask_h5loc !")
        mask_file_ = os.path.join(output_folder, ".mask_temp.npy")
        config['make_detector|in_mask_file'] = mask_file_
    
    # set up new project
    emc.new_project(data_path = input_h5, \
                         inh5 = data_loc, \
                         path = output_folder, \
                         name = proj_name)

    # configure
    emc.config(config)
    try:
        os.remove(mask_file_)
    except:
        pass

    # dry-run
    emc.run(num_proc=8, num_thread=12, iters=30, nohup=False, resume=False, cluster=True)
    print("(Look into 'submit_job.sh' and change the arguments of emc executables.)")