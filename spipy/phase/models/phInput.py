import numpy as np
from .model_interface import PhModel, streamData
from . import model_utils


class phInput(PhModel):

    def __init__(self, config_dict, name=None, data_reload=None):
        # config
        self.config_dict = config_dict
        self.data_reload = {}
        # data_reload = {"pattern" : ..., "mask" : ..., "initial" : ...}
        self.reload(data_reload)
        # node name
        if name is None:
            name = "Input"
        else:
            name = str(name)
        super().__init__(name)
        # config bk
        self.config_bk["config_dict"] = self.config_dict.copy()

    def reload(self, new_data):
        # new_data = {"pattern" : ..., "mask" : ..., "initial" : ...}
        new_data_template = ["pattern", "mask", "initial"]
        if new_data is not None:
            for k, v in new_data.items():
                if k not in new_data_template:
                    print("[WARN] data_reload unknown key : %s" % k)
                self.data_reload[k] = v

    def after(self, fatherobj):
        raise RuntimeError("phInput node doesn't have father !")

    def run(self, datapack=None, rank=0):
        '''config_dict = 
            { 
            "pattern_path" : xxx.npy,
            "mask_path" : xxx.npy, (1 is masked area)
            "center_mask" : 5,
            "edge_mask" : [60,64],
            "subtract_percentile" : False,
            "fixed_support_r" : 20,
            "background" : True,
            "initial_model" : xxx.npy
            } 
        '''
        '''data_reload
            {
            "pattern" : list (pattern intensity)
            "mask" : list (mask area)
            "initial" : list (initial sample model inensity)
            }
        '''
        if rank == 0 : print("%10s : Configuration ..." % (self.name+"-"+str(self.id)))
        pattern = None
        good_pixel = None
        background = None
        sample_ret = None
        fixed_support = None
        # pattern
        if 'pattern' in self.data_reload.keys():#self.config_dict.keys():
            pattern = np.array(self.data_reload['pattern'])
        else:
            pattern = np.load(self.config_dict['pattern_path'])
        pattern = np.nan_to_num(pattern)
        pattern[pattern < 0] = 0
        good_pixel = np.ones(pattern.shape, dtype=int)
        # center
        center = (np.array(pattern.shape)-1)/2.0
        # usermask
        if 'mask' in self.data_reload.keys():#self.config_dict.keys():
            usermask = np.array(self.data_reload['mask'], dtype=int)
        elif self.config_dict['mask_path'] is not None:
            usermask = np.load(self.config_dict['mask_path'])
        else:
            usermask = None
        if usermask is not None:
            if usermask.shape != pattern.shape:
                raise RuntimeError("Input user mask has different size with input pattern !")
            good_pixel[np.where(usermask==1)] = 0
        # center mask
        if self.config_dict['center_mask'] is not None and self.config_dict['center_mask']>0:
            hole = model_utils.make_holes(pattern.shape, center, self.config_dict['center_mask'])
            good_pixel = good_pixel & (~hole)
        # edge mask
        if self.config_dict['edge_mask'] is not None:
            hole1 = model_utils.make_holes(pattern.shape, center, self.config_dict['edge_mask'][0])
            hole2 = model_utils.make_holes(pattern.shape, center, self.config_dict['edge_mask'][1])
            hole = hole2 & (~hole1)
            good_pixel = good_pixel & (~hole)
        # edge remove
        if self.config_dict['edge_remove'] is not None:
            hole1 = model_utils.make_holes(pattern.shape, center, self.config_dict['edge_remove'][0])
            hole2 = model_utils.make_holes(pattern.shape, center, self.config_dict['edge_remove'][1])
            hole = hole2 & (~hole1)
            pattern *= (~hole)
        # fixed support
        if self.config_dict['fixed_support_r'] is not None and self.config_dict['fixed_support_r']>0:
            fixed_support = model_utils.make_holes(pattern.shape, center, self.config_dict['fixed_support_r'])
            fixed_support = fixed_support.astype(int)
        # background
        if self.config_dict['background']:
            background = np.random.random(size=pattern.shape)
        else:
            background = None
        # sample_ret
        if self.config_dict['initial_model'] is not None:
            if 'initial' in self.data_reload.keys():#self.config_dict.keys():
                sample_ret = np.array(self.data_reload['initial'])
            else:
                sample_ret = np.load(self.config_dict['initial_model'])
            if sample_ret.shape != pattern.shape:
                raise RuntimeError("Initial model has different size with input pattern !")
        else:
            sample_ret = np.random.random(size=pattern.shape)
        # subtract
        if self.config_dict['subtract_percentile']>0 and self.config_dict['subtract_percentile']<1:
            p = np.percentile(pattern[(good_pixel > 0) * (pattern > 0)], self.config_dict['subtract_percentile'])
            pattern -= p
            pattern[pattern < 0] = 0
        # fftshift
        pattern = np.fft.fftshift(pattern)
        good_pixel = np.fft.fftshift(good_pixel)
        sample_ret = np.fft.fftshift(sample_ret)
        if fixed_support is not None:
            fixed_support = np.fft.fftshift(fixed_support)
        if background is not None:
            background = np.fft.fftshift(background)
        # make input
        self.datapack = streamData(sample_ret, pattern, fixed_support, background, good_pixel)
        self.datapack.add_metrics(self.name, self.id, [], [])
        return self.datapack

    def change_dataset(self, new_data_dict):
        # change data files
        '''new_data_dict
        {
        "pattern_path" : xxx.npy,
        "mask_path" : xxx.npy, (1 is masked area)
        "initial_model" : xxx.npy
        }
        '''
        for k, v in new_data_dict.items():
            if k in self.config_dict.keys():
                self.config_dict[k] = v
            else:
                raise ValueError("Key '%s' is not a configuration parameter" % str(k))








