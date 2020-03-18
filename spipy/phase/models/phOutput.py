import numpy as np
from .model_interface import PhModel, streamData
from . import model_utils, model_merge


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
        self.eMods = {}  # {repeat_0:[[...], [...], ...], ...}
        self.eCons = {}  # same with eMods
        self.stream_path = []
        self.output = {}
        # no config dict

    def __add_child(self, childobj):
        raise RuntimeError("[Error] phOutput node cannot have children !")

    def run(self, datapack, rank=0):
        # append solutions of this rank
        if datapack.background is None:
            self.backgrounds.append(0)
        else:
            self.backgrounds.append(datapack.background.copy())
        self.sample_rets.append(datapack.sample_ret.copy())
        self.supports.append(datapack.support.copy())
        # deal with eMods and eCons
        if self.repeat_id not in self.eMods.keys():
            self.eMods[self.repeat_id] = datapack.err_mod
            self.eCons[self.repeat_id] = datapack.err_con
        else:
            self.eMods[self.repeat_id].extend(datapack.err_mod)
            self.eCons[self.repeat_id].extend(datapack.err_con)
        # deal with stream_path
        if self.repeat_id == 0:
            self.stream_path.extend(datapack.stream_path)
        return datapack

    def merge(self, msize=1):
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
        self.output['eMod'] = []
        self.output['eCon'] = []
        self.output['stream_path'] = []
        # deal with eCons and eMods and stream path
        for emod in self.eMods.values():
            if len(self.output['eMod']) == 0:
                self.output['eMod'] = emod
            else:
                for i, part in enumerate(emod):
                    self.output['eMod'][i] = np.array(part) + np.array(self.output['eMod'][i])
        for econ in self.eCons.values():
            if len(self.output['eCon']) == 0:
                self.output['eCon'] = econ
            else:
                for i, part in enumerate(econ):
                    self.output['eCon'][i] = np.array(part) + np.array(self.output['eCon'][i])
        self.output['stream_path'] = self.stream_path.copy()
        return self.output