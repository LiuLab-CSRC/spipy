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
            "center" : [61,61],
            "center_mask" : 5,
            "edge_mask" : None,
            "subtract_percentile" : False,
            "fixed_support_r" : None,
            "background" : True,
            "initial_model" : None
        }
        iters = [200,100,100,200]
        support_size = 100
        beta = 0.8
        gamma = 0.05
        newdataset = {"pattern_path" : "pattern.npy", "mask_path" : "pat_mask.npy", "initial_model" : None}

    else:

        config_input = {
            "pattern_path" : "volume.npy",
            "mask_path" : None,
            "center" : [62,62,62],
            "center_mask" : 5,
            "edge_mask" : [64,70],
            "subtract_percentile" : False,
            "fixed_support_r" : None,
            "background" : True,
            "initial_model" : None
        }
        iters = [100,50,50,100]
        support_size = 2000
        beta = 0.8
        gamma = 0.5
        newdataset = {"pattern_path" : "volume.npy", "mask_path" : None, "initial_model" : None}

    l1 = phmodel.pInput(config_input)
    l2 = phmodel.HIO(iters[0], support_size, gamma).after(l1)
    l3 = phmodel.RAAR(iters[1], support_size, beta).after(l2)
    l4 = phmodel.DM(iters[2], support_size).after(l3)
    l5 = phmodel.ERA(iters[3], support_size).after(l4)
    l6 = phmodel.pOutput().after(l5)

    runner = phexec.Runner(inputnode = l1, outputnode = l6)
    out = runner.run(repeat = 1)
    
    if mrank == 0:
        runner.plot_result(out)


    # Dump model and load it

    runner.dump_model("temp_model.json", skeleton=False)
    runner2 = phexec.Runner(inputnode = None, outputnode = None, \
                            loadfile = "temp_model.json", change_dataset = None)
    out = runner2.run(repeat = 1)


