import numpy as np
from .model_interface import PhModel, streamData
from . import model_utils

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()


class HIO(PhModel):

    def __init__(self, iteration, support_size, gamma, name=None):
        # config
        self.iteration = int(iteration)
        self.sup_size = int(support_size)
        self.gamma = float(gamma)
        # buffer
        self.bg_av = None
        self.support = None
        # node name
        if name is None:
            name = "HIO"
        else:
            name = str(name)
        super().__init__(name)
        # config dict
        self.config_bk = {"name" : self.name, "iteration" : self.iteration, "support_size" : self.sup_size, "gamma" : self.gamma}

    def run(self, datapack):

        # going into iterations
        for i in range(self.iteration):

            sample_ret_0 = datapack.copy_sample()

            model_utils.module_projection(datapack)

            self.radial_s = model_utils.support_projection(datapack, self.sup_size, self.radial_s)

            unsup_index = np.where(datapack.support < 1)

            datapack.sample_ret[unsup_index] = sample_ret_0[unsup_index] * (1 - self.gamma)

            eMod = datapack.calc_eMod()
            eCon = datapack.calc_eCon(sample_ret_0)
            datapack.add_metrics(eMod, eCon)

            if rank == 0 : self.show_progress(i, self.iteration, eCon, eMod )

        return datapack