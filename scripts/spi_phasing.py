import sys
import os
import shutil
import argparse
import numpy as np
import h5py
from spipy.phase import phexec, phmodel
from mpi4py import MPI

comm = MPI.COMM_WORLD
m_rank = comm.Get_rank()
m_size = comm.Get_size()

if __name__ == '__main__':

    # parse cmd arguments
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description = "Phase retrieval of single-particle diffraction patterns ( Y=|FFT(X)|.^2 ).")
    parser.add_argument("-d", "--data_file", type=str, help="Input pattern file, support npy and HDF5 file format.", required=True)
    parser.add_argument("-o", "--output", type=str, help="Output result file, HDF5 format.", required=True)
    parser.add_argument("-a", "--algorithms", type=str, help="Phasing algorithms, use comma to separate.", required=True)
    parser.add_argument("-n", "--iterations", type=str, help="Number of iterations for each algorithm, use comma to separate.", required=True)
    parser.add_argument("-s", "--support_size", type=int, help="Estimation of the number of pixels inside retrieved support.", required=True)
    parser.add_argument("--stacking", action="store_true", default=False, help="Set to tell the program that the input data are multi-frame data.")
    parser.add_argument("--mask_file", type=str, default="none", help="Input mask file, which stores one mask, and masked pixels have the value of 1. Support npy and HDF5 file format, default is none.")
    parser.add_argument("--repeat", type=int, default=1, help="Repetition time of independent phasing using same parameters, default=1.")
    parser.add_argument("--center_mask", type=float, default=0, help="The radius of a hole area at the center of pattern to be masked, default is 0.")
    parser.add_argument("--edge_mask", type=str, default="none", help="The radii of a ring area of pattern to be masked, e.g 'r0,r1', default is none.")
    parser.add_argument("--fixed_support_r", type=float, default=0, help="The radius of a fixed circle support, default is 0.")
    parser.add_argument("--background", action="store_true", default=False, help="Set to do fitting on background scattering.")
    parser.add_argument("--initial_model", type=str, default="none", help="The initial model of retrieved sample (.npy file), default is 'none' and use random initiation.")
    parser.add_argument("--beta", type=float, default=0.8, help="Beta value of RAAR algorithm, default=0.8.")
    parser.add_argument("--gamma", type=float, default=0.5, help="Gamma value of HIO and HPR algorithms, default=0.5.")
    parser.add_argument("--data_h5loc", type=str, default="none", help="If --data_file is in HDF5 format, please provide the location of pattern inside h5 file.")
    parser.add_argument("--mask_h5loc", type=str, default="none", help="If --mask_file is in HDF5 format, please provide the location of mask inside h5 file.")
    parser.add_argument("--paramode", type=str, choices=['reptpara','datapara'], default="reptpara", help="Parallel mode, 'reptpara' means parallel on repetition, 'datapara' means parallel on dataset, default is 'reptpara'.")
    parser.add_argument("--save_phaser", type=str, default="none", help="Give a json file to save this phasing network archetecture, default is none.")
    parser.add_argument("-j", type=int, default=1, help="Number of processes, default=1.")
    args = parser.parse_args()

    # get parameters
    config_input = {
        "pattern_path" : None,
        "mask_path" : None,
        "center_mask" : args.center_mask,
        "edge_mask" : None,
        "edge_remove" : None,
        "subtract_percentile" : False,
        "fixed_support_r" : args.fixed_support_r,
        "background" : args.background,
        "initial_model" : None
    }
    data_reload = {
        "pattern" : None,
        "mask" : None
    }
    pattern_set = None
    mask = None
    pattern_serial = None
    support_size = args.support_size
    iterations = list(map(int, args.iterations.split(',')))
    algorithms = args.algorithms.split(',')
    gamma = args.gamma
    beta = args.beta

    # patterns
    if os.path.splitext(args.data_file)[1] == ".h5":
        with h5py.File(args.data_file, 'r') as fp:
            if args.stacking:
                if args.paramode == "datapara":
                    num_pat = fp[args.data_h5loc].shape[0]
                    tmplin = np.linspace(0, num_pat, m_size+1, dtype=int)
                    my_start = tmplin[m_rank]
                    my_end = tmplin[m_rank+1]
                    pattern_set = fp[args.data_h5loc][my_start:my_end]
                else:
                    pattern_set = fp[args.data_h5loc][()]
            else:
                pattern_set = [fp[args.data_h5loc][()]]
    else:
        pattern_set = np.load(args.data_file)
        if args.stacking:
            if args.paramode == "datapara":
                num_pat = pattern_set.shape[0]
                tmplin = np.linspace(0, num_pat, m_size+1)
                my_start = tmplin[m_rank]
                my_end = tmplin[m_rank+1]
                pattern_set = pattern_set[my_start:my_end]
        else:
            pattern_set = [pattern_set]
    pattern_serial = args.data_file + "###" + args.data_h5loc

    # mask
    if args.mask_file.upper() != "NONE":
        if os.path.splitext(args.mask_file)[1] == ".h5":
            with h5py.File(args.mask_file, 'r') as fp:
                mask = fp[args.mask_h5loc][()]
        else:
            mask = np.load(args.mask_file)
    data_reload["mask"] = mask

    # check data stacking
    if len(mask.shape) > 3 or ( args.stacking and len(mask.shape) == len(pattern_set.shape) ):
        if m_rank == 0: raise ValueError("Input mask file should only contain ONE mask (stacked data not allowed) !")
        MPI.Finalize()
        sys.exit(1)
    if args.paramode == "datapara" and not args.stacking:
        if m_rank == 0: raise ValueError("Stacked input data is required for 'datapara' mode !")
        MPI.Finalize()
        sys.exit(1)

    # check output path
    if not os.path.isdir(os.path.dirname(os.path.abspath(args.output))):
        if m_rank == 0: raise ValueError("Output folder is invalid !")
        MPI.Finalize()
        sys.exit(1)
    if os.path.exists(args.output):
        if m_rank == 0: raise ValueError("Output file already exists !")
        MPI.Finalize()
        sys.exit(1)
    if args.save_phaser.upper() != "NONE":
        if not os.path.isdir(os.path.dirname(os.path.abspath(args.save_phaser))):
            if m_rank == 0: raise ValueError("The file path to save phasing network is invalid !")
            MPI.Finalize()
            sys.exit(1)

    # check iterations
    if len(iterations) != len(algorithms):
        if m_rank == 0: raise ValueError("The iterations and algorithms are corresponding one by one !")
        MPI.Finalize()
        sys.exit(1)

    comm.Barrier()

    # others
    if args.edge_mask.upper() != "NONE":
        config_input["edge_mask"] = list(map(float, args.edge_mask.split(',')))
    if args.initial_model.upper() != "NONE":
        config_input["initial_model"] = args.initial_model

    # build phasing network
    l_in = phmodel.pInput(config_input)
    l = l_in
    for algo, iter in zip(algorithms, iterations):
        if algo in ["HIO", "HPR"]:
            l = eval("phmodel.%s"%algo)(iter, support_size, gamma).after(l)
        elif algo == "RAAR":
            l = phmodel.RAAR(iter, support_size, beta).after(l)
        elif algo in ["DM", "ERA"]:
            l = eval("phmodel.%s"%algo)(iter, support_size).after(l)
        else:
            if m_rank == 0: raise RuntimeError("Unknown algorithm %s" % algo)
            MPI.Finalize()
            sys.exit(1)
    l_out = phmodel.pOutput().after(l)

    # do phasing
    if args.paramode == "datapara":

        runner = phexec.Runner(inputnodes = [l_in], outputnode = l_out, comm = None)
        runner.rank = m_rank
        repeat = args.repeat
        savefile = os.path.splitext(args.output)[0] + "-part%03d" % m_rank + ".h5"

        for i, pattern in enumerate(pattern_set):
            pat_id = i + my_start
            data_reload["pattern"] = pattern
            l_in.reload(data_reload)
            try:
                out = runner.run(repeat = repeat)
            except Exception as err:
                print(err)
                sys.exit(1)
            # Write to h5. This is not save as different ranks are treating with
            # different data. If one is down the other will get stuck.
            with h5py.File(savefile, 'a') as fp:
                for k, v in out.items():
                    try:
                        fp.create_dataset("pat-%d/%s"%(pat_id, k), data=v, chunks=True, compression="gzip")
                    except:
                        print("[WARN] Skip writting '%s' values : %s" %(k, v))
            # reset phaser data
            runner.reset_network()

        with h5py.File(savefile, 'a') as fp:
            fp.create_dataset("information", data=h5py.Empty(int))
            fp["information"].attrs["dataset_path"] = pattern_serial
            fp["information"].attrs["cmd_line"] = " ".join(sys.argv)

    else:

        if m_size > args.repeat:
            if m_rank == 0: raise RuntimeError("The repeat times are too less to assign a task to every rank.")
            MPI.Finalize()
            sys.exit(1)

        runner = phexec.Runner(inputnodes = [l_in], outputnode = l_out, comm = comm)
        tmplin = np.linspace(0, args.repeat, m_size+1, dtype=int)
        repeat = tmplin[m_rank+1] - tmplin[m_rank]
        savefile = args.output

        for i, pattern in enumerate(pattern_set):
            comm.Barrier()
            pat_id = i
            data_reload["pattern"] = pattern
            l_in.reload(data_reload)
            try:
                out = runner.run(repeat = repeat)
            except Exception as err:
                print(err)
                sys.exit(1)
            # Write to h5. This is not save as only rank-0 will go through
            # the following procedure. If rank-0 is down the other will get stuck.
            if m_rank == 0:
                with h5py.File(savefile, 'a') as fp:
                    for k, v in out.items():
                        try:
                            fp.create_dataset("pat-%d/%s"%(pat_id, k), data=v, chunks=True, compression="gzip")
                        except:
                            print("[WARN] Skip writting '%s' values : %s" %(k, v))
            # reset phaser data
            runner.reset_network()
        
        if m_rank == 0:
            with h5py.File(savefile, 'a') as fp:
                fp.create_dataset("information/dataset_path", data=pattern_serial)
                fp.create_dataset("information/cmd_line", data=" ".join(sys.argv))

    # dump model
    if m_rank == 0 and args.save_phaser.upper() != "NONE":
        runner.dump_model(args.save_phaser, skeleton=True)

    MPI.Finalize()