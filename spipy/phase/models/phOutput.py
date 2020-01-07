import numpy as np
from .model_interface import PhModel, streamData
from . import model_utils, model_merge

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
msize = comm.Get_size()


class phOutput(PhModel):

    def __init__(self, name=None):
        # node name
        if name is None:
            name = "Output"
        else:
            name = str(name)
        super().__init__(name)
        self.backgrounds = []
        self.sample_rets = []
        self.supports = []
        self.eMods = []
        self.eCons = []
        self.output = {}

    def run(self, datapack):
        # append solutions of this rank
        if datapack.background is None:
            self.backgrounds.append(0)
        else:
            self.backgrounds.append(datapack.background)
        self.sample_rets.append(datapack.sample_ret)
        self.supports.append(datapack.support)
        self.eMods.append(datapack.err_mod)
        self.eCons.append(datapack.err_con)
        return datapack

    def merge(self):
        # merge solutions of this rank, N repeats
        if msize == 1:
            silence = False
        else:
            silence = True
        this_sample_ret, PRTF = model_merge.merge_sols(np.array(self.sample_rets), silence)
        this_support, _ = model_merge.merge_sols(np.array(self.supports), True)
        this_background = np.mean(self.backgrounds, axis=0)
        self.output['sample_ret'] = this_sample_ret
        self.output['support'] = this_support
        self.output['PRTF'] = PRTF
        self.output['background'] = this_background
        self.output['eMod'] = np.mean(self.eMods,axis=0)
        self.output['eCon'] = np.mean(self.eCons,axis=0)
        return self.output