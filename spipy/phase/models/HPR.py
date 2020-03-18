import numpy as np
from .model_interface import PhModel, streamData
from . import model_utils


class HPR(PhModel):

    def __init__(self, iteration, support_size, gamma, hprcoff=0.5, name=None):
        # config
        self.iteration = int(iteration)
        self.sup_size = int(support_size)
        self.gamma = float(gamma)
        self.hprcoff = float(hprcoff)
        # buffer
        self.bg_av = None
        self.support = None
        # node name
        if name is None:
            name = "HPR"
        else:
            name = str(name)
        super().__init__(name)
        # config dict
        self.config_bk["iteration"] = self.iteration
        self.config_bk["support_size"] = self.sup_size
        self.config_bk["gamma"] = self.gamma
        self.config_bk["hprcoff"] = self.hprcoff

    def __overlap(self, sup0, sup1):
        a = (sup0>0) & (sup1>0)
        b = (sup0>0) | (sup1>0)
        return float(np.sum(a))/np.sum(b)

    def run(self, datapack, rank=0):
        import matplotlib.pyplot as plt
        err_con = []
        err_mod = []

        # going into iterations
        for i in range(self.iteration):

            sample_ret_0 = datapack.copy_sample()
            support_0 = datapack.copy_support()

            model_utils.module_projection(datapack)

            RM = np.abs(datapack.sample_ret) * (1 + self.gamma) - np.abs(sample_ret_0)

            self.radial_s = model_utils.support_projection(datapack, self.sup_size, self.radial_s)

            if self.__overlap(support_0, datapack.support) > self.hprcoff:
                unsup_index = np.where((datapack.support < 1) | (RM < 0))
            else:
                unsup_index = np.where((datapack.support < 1))

            datapack.sample_ret[unsup_index] = (1 - self.gamma) * sample_ret_0[unsup_index] + self.gamma * datapack.sample_ret[unsup_index]

            eMod = datapack.calc_eMod()
            eCon = datapack.calc_eCon(sample_ret_0)
            err_con.append(eCon)
            err_mod.append(eMod)

            if rank == 0 : self.show_progress(i, self.iteration, eCon, eMod )

        datapack.add_metrics(self.name, self.id, err_mod, err_con)

        return datapack