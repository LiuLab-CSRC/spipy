import numpy as np
from .model_interface import PhModel
from . import model_utils


class RAAR(PhModel):

    def __init__(self, iteration, support_size, beta, name=None):
        # config
        self.iteration = int(iteration)
        self.sup_size = int(support_size)
        self.beta = float(beta)
        # buffer
        self.bg_av = None
        self.support = None
        # node name
        if name is None:
            name = "RAAR"
        else:
            name = str(name)
        super().__init__(name)
        # config dict
        self.config_bk["iteration"] = self.iteration
        self.config_bk["support_size"] = self.sup_size
        self.config_bk["beta"] = self.beta

    def run(self, datapack, rank=0):
        err_con = []
        err_mod = []
        node = datapack.dump_node()

        # going into iterations
        for i in range(self.iteration):

            node_0 = node.copy()
            
            node_superr, self.radial_s, _, _ = model_utils.support_projection_node\
                                                    (datapack, node, self.sup_size, self.radial_s)
            
            node_superr = node_superr * 2 - node_0

            node_perr = model_utils.module_projection_node(datapack, node_superr) * 2 - node_superr

            node = (node_perr * 0.5 + node_0 * 0.5) * self.beta + model_utils.module_projection_node(datapack, node_0) * (1-self.beta)
            
            node, self.radial_s, self.bg_av, self.support = model_utils.support_projection_node\
                                                    (datapack, node, self.sup_size, self.radial_s)
            
            eCon = datapack.calc_eCon(node_0.copy_sample(), node)
            eMod = datapack.calc_eMod(node)
            err_con.append(eCon)
            err_mod.append(eMod)

            if rank == 0 : self.show_progress(i, self.iteration, eCon, eMod )

        datapack.load_node(node)
        if self.bg_av is not None:
            datapack.background_av = self.bg_av.copy()
        datapack.support = self.support.copy()
        datapack.add_metrics(self.name, self.id, err_mod, err_con)

        return datapack