import sys
import numpy as np
from spipy.phase import phexec, phmodel

from mpi4py import MPI
comm = MPI.COMM_WORLD
mrank = comm.Get_rank()
msize = comm.Get_size()

if __name__ == "__main__":
    
    try:
        task = int(sys.argv[1])
    except:
        print("Usage : python test_phmodel.py [2/3]")
        sys.exit(0)

    if task == 2:

        config_input = {
            "pattern_path" : "pattern.npy",
            "mask_path" : "pat_mask.npy",
            "center_mask" : 3,
            "edge_mask" : [64,70],
            "edge_remove" : None,
            "subtract_percentile" : False,
            "fixed_support_r" : 20,
            "background" : True,
            "initial_model" : None
        }
        iters = [200,100,100]
        support_size = 85
        beta = 0.8
        gamma = 0.1
        newdataset = {"pattern_path" : "pattern.npy", "mask_path" : "pat_mask.npy", "initial_model" : None}

    else:

        config_input = {
            "pattern_path" : "volume.npy",
            "mask_path" : None,
            "center_mask" : 5,
            "edge_mask" : [64,70],
            "edge_remove" : None,
            "subtract_percentile" : False,
            "fixed_support_r" : None,
            "background" : True,
            "initial_model" : None
        }
        iters = [100,100,100]
        support_size = 2000
        beta = 0.8
        gamma = 0.3
        newdataset = {"pattern_path" : "volume.npy", "mask_path" : None, "initial_model" : None}

    l1_0 = phmodel.pInput(config_input)
    l1_1 = phmodel.pInput(config_input)
    l2_0 = phmodel.HIO(iters[0], support_size, gamma).after(l1_0)
    l2_1 = phmodel.HPR(iters[0], support_size, gamma, 0.5).after(l1_1)
    l2_2 = phmodel.RAAR(iters[0], support_size, beta).after(l1_1)
    lm = phmodel.pMerge().after(l2_0).after(l2_1).after(l2_2)
    l4 = phmodel.DM(iters[1], support_size).after(lm)
    l5 = phmodel.ERA(iters[2], support_size).after(l4)
    l6 = phmodel.pOutput().after(l5)

    runner = phexec.Runner(inputnodes = [l1_0, l1_1], outputnode = l6, comm = comm)
    out = runner.run(repeat = 2)

    if mrank == 0:
        runner.plot_result(out)

    # Dump model
    runner.dump_model("temp_model.json", skeleton=False)
'''
    # Reload model
    runner2 = phexec.Runner(inputnodes = None, outputnode = None, \
                loadfile = "temp_model.json", reload_dataset = {0:newdataset}, comm = comm)
    out = runner2.run(repeat = 1)
'''

