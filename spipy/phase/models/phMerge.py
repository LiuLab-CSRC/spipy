import numpy as np
from .model_interface import PhModel, streamData
from . import model_utils, model_merge

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()


class phMerge(PhModel):

    def __init__(self, name=None):
        # config
        self.backgrounds = []
        self.sample_rets = []
        self.supports = []
        self.eMods = []
        self.eCons = []
        self.stream_path = []
        # node name
        if name is None:
            name = "Merge"
        else:
            name = str(name)
        super().__init__(name)
        # no config dict
    '''
    def after(self, fatherobj):
        fatherobj.__add_child(self)
        self.father_num += 1
        return self
    '''
    def run(self, datapack):
        # append solutions of this rank
        if datapack.background is None:
            self.backgrounds.append(0)
        else:
            self.backgrounds.append(datapack.background.copy())
        self.sample_rets.append(datapack.sample_ret.copy())
        self.supports.append(datapack.support.copy())
        # stream variables
        self.eCons.extend(datapack.err_con)
        self.eMods.extend(datapack.err_mod)
        self.stream_path.extend(datapack.stream_path)
        # if all fathers are appended
        if len(self.sample_rets) < self.father_num:
            if rank == 0 : print("%10s : Merge data stream (Barrier)" % (self.name+"-"+str(self.id)))
            return None
        else:
            this_sample_ret, _ = model_merge.merge_sols(np.array(self.sample_rets), True)
            this_support, _ = model_merge.merge_sols(np.array(self.supports), True)
            this_background = np.mean(self.backgrounds, axis=0)
            datapack.sample_ret = this_sample_ret
            datapack.support = this_support
            if datapack.background is not None:
                datapack.background = this_background
            # deal with eMods, eCons and stream_path
            datapack.err_mod = self.eCons.copy()
            datapack.err_con = self.eMods.copy()
            datapack.stream_path = self.stream_path.copy()
            datapack.add_metrics(self.name, self.id, [], [])
            # clear buffer and return
            self.sample_rets.clear()
            self.supports.clear()
            self.backgrounds.clear()
            self.eMods.clear()
            self.eCons.clear()
            self.stream_path.clear()
            if rank == 0 : print("%10s : Merge data stream (Done)" % (self.name+"-"+str(self.id)))
            return datapack