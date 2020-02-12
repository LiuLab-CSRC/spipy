import numpy as np
from .model_interface import PhModel, streamData
from . import model_utils

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()


class ERA(PhModel):

    def __init__(self, iteration, support_size, name=None):
        # config
        self.iteration = int(iteration)
        self.sup_size = int(support_size)
        # node name
        if name is None:
            name = "ERA"
        else:
            name = str(name)
        super().__init__(name)
        # config dict
        self.config_bk["iteration"] = self.iteration
        self.config_bk["support_size"] = self.sup_size

    def run(self, datapack):
        err_con = []
        err_mod = []

        # going into iterations
        for i in range(self.iteration):

            sample_ret_0 = datapack.copy_sample()

            model_utils.module_projection(datapack)

            self.radial_s = model_utils.support_projection(datapack, self.sup_size, self.radial_s)

            eMod = datapack.calc_eMod()
            eCon = datapack.calc_eCon(sample_ret_0)
            err_con.append(eCon)
            err_mod.append(eMod)

            if rank == 0 : self.show_progress(i, self.iteration, eCon, eMod )

        datapack.add_metrics(self.name, self.id, err_mod, err_con)

        return datapack