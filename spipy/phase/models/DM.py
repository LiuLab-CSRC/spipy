import numpy as np
from .model_interface import PhModel
from . import model_utils

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()


class DM(PhModel):

    def __init__(self, iteration, support_size, name=None):
        # config
        self.iteration = int(iteration)
        self.sup_size = int(support_size)
        # buffer
        self.bg_av = None
        self.support = None
        # node name
        if name is None:
            name = "DM"
        else:
            name = str(name)
        super().__init__(name)
        # config dict
        self.config_bk = {"name" : self.name, "iteration" : self.iteration, "support_size" : self.sup_size}

    def run(self, datapack):

        node_0 = datapack.dump_node()
        node_sup, self.radial_s, _, _ = model_utils.support_projection_node(datapack, node_0, self.sup_size, self.radial_s)

        # going into iterations
        for i in range(self.iteration):

            sample_ret_0 = node_0.copy_sample()

            node_0 += model_utils.module_projection_node(datapack, node_sup * 2 - node_0) - node_sup

            node_sup, self.radial_s, self.bg_av, self.support = model_utils.support_projection_node\
                                                    (datapack, node_0, self.sup_size, self.radial_s)

            eCon = datapack.calc_eCon(sample_ret_0, node_0)
            eMod = datapack.calc_eMod(node_sup)
            datapack.add_metrics(eMod, eCon)

            if rank == 0 : self.show_progress(i, self.iteration, eCon, eMod )

        datapack.load_node(node_sup)
        if self.bg_av is not None:
            datapack.background_av = self.bg_av.copy()
        datapack.support = self.support.copy()

        return datapack


