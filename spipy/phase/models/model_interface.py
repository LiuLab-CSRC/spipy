import sys
import numpy as np

class streamData(object):

    def __init__(self, sample_ret, diffraction_pat, init_support, background, good_pixel):
        self.sample_ret = sample_ret.copy()
        self.diffraction_amp = np.sqrt(diffraction_pat)
        if init_support is not None:
            self.init_support = init_support.copy()
            self.support = init_support.copy()
        else:
            self.init_support = None
            self.support = np.ones(self.diffraction_amp.shape)
        if background is not None:
            self.background = background.copy()
        else:
            self.background = None
        if good_pixel is None or type(good_pixel) in [int, float]:
            self.good_pixel = 1
        else:
            self.good_pixel = good_pixel.copy()
        self.background_av = None
        self.I_norm = (self.good_pixel * diffraction_pat).sum()
        # data stream path
        self.stream_path = []
        self.err_mod = []
        self.err_con = []

    def copy(self):
        new_stream = streamData(self.sample_ret, self.diffraction_amp**2, \
                        self.init_support, self.background, \
                        self.good_pixel)
        new_stream.stream_path = self.stream_path.copy()
        new_stream.err_mod = self.err_mod.copy()
        new_stream.err_con = self.err_con.copy()
        return new_stream

    def copy_sample(self):
        return self.sample_ret.copy()

    def copy_diff_amp(self):
        return self.diffraction_amp.copy()

    def dump_node(self):
        return nodeData(self)

    def load_node(self, node):
        self.sample_ret = node.data['sample_ret'].copy()
        if 'background' in node.data.keys():
            self.background = node.data['background'].copy()
        else:
            self.background = None

    def add_metrics(self, name, id, eMods, eCons):
        assert len(eMods) == len(eCons), "[Error] Fatal bug, please report."
        key = "%s-%d" % (name, id)
        self.stream_path.append(key)
        self.err_mod.append(eMods)
        self.err_con.append(eCons)

    def calc_intensity(self, node=None):
        if node is None:
            amp = np.fft.fftn(self.sample_ret)
            bg = self.background
        else:
            amp = np.fft.fftn(node.data['sample_ret'])
            if 'background' in node.data.keys():
                bg = node.data['background']
            else:
                bg = None
        if bg is not None:
            I = (amp.conj() * amp).real + bg**2
        else:
            I = (amp.conj() * amp).real
        return I

    def calc_eMod(self, node=None):
        M      = self.calc_intensity(node)
        eMod   = np.sum( self.good_pixel * ( np.sqrt(M) - self.diffraction_amp )**2 )
        eMod   = np.sqrt( eMod / self.I_norm )
        return eMod

    def calc_eCon(self, sample_ret_0, node=None):
        if node is None:
            delta = self.sample_ret - sample_ret_0
        else:
            delta = node.data['sample_ret'] - sample_ret_0
        eCon = np.sum( (delta*delta.conj()).real ) / np.sum( (sample_ret_0*sample_ret_0.conj()).real )
        return np.sqrt(eCon)


class nodeData(object):

    def __init__(self, Data_buf):
        self.data = {}
        if type(Data_buf) == nodeData:
            for k in Data_buf.data.keys():
                if Data_buf.data[k] is not None:
                    self.data[k] = Data_buf.data[k].copy()             
        else:
            self.data['sample_ret'] = Data_buf.sample_ret.copy()
            if Data_buf.background is not None:
                self.data['background'] = Data_buf.background.copy()

    def __add__(self, item):
        out = nodeData(self)
        for k in self.data.keys():
            if type(item) == nodeData:
                out.data[k] = self.data[k] + item.data[k]
            else:
                out.data[k] = self.data[k] + item
        return out

    def __iadd__(self, item):
        for k in self.data.keys():
            if type(item) == nodeData:
                self.data[k] += item.data[k]
            else:
                self.data[k] += item
        return self

    def __sub__(self, item):
        out = nodeData(self)
        for k in self.data.keys():
            if type(item) == nodeData:
                out.data[k] = self.data[k] - item.data[k]
            else:
                out.data[k] = self.data[k] - item
        return out

    def __isub__(self, item):
        for k in self.data.keys():
            if type(item) == nodeData:
                self.data[k] -= item.data[k]
            else:
                self.data[k] -= item
        return self

    def __mul__(self, item):
        out = nodeData(self)
        for k in self.data.keys():
            if type(item) == nodeData:
                out.data[k] = self.data[k] * item.data[k]
            else:
                out.data[k] = self.data[k] * item
        return out

    def __imul__(self, item):
        for k in self.data.keys():
            if type(item) == nodeData:
                self.data[k] *= item.data[k]
            else:
                self.data[k] *= item
        return self

    def copy(self):
        return nodeData(self)

    def copy_sample(self):
        return self.data['sample_ret'].copy()

    def add_data(self, key, value):
        self.data[key] = value
        



class PhModel(object):

    __classname__ = ""
    __globalID__ = 0

    def __init__(self, name):
        self.name = name
        self.id = PhModel.__globalID__
        self.father_num = 0
        self.repeat_id = 0
        self.children = []  # multi-children
        self.radial_s = None
        self.config_bk = {"name" : self.name}
        PhModel.__globalID__ += 1

    def __add_child(self, childobj):
        if childobj in self.children:
            raise RuntimeError("[Error] One child cannot be added twice !")
        self.children.append(childobj)

    def set_repeat(self, repeat_id):
        self.repeat_id = repeat_id

    def has_children(self):
        if len(self.children) > 0 : return True
        else : return False

    def rename(self, newname):
        self.name = newname
        self.config_bk["name"] = newname

    def after(self, fatherobj):
        fatherobj.__add_child(self)
        self.father_num += 1
        return self

    def show_progress(self, iter_now, iter_all, emod, esup, barLength = 10):
        progress = float(iter_now+1) / float(iter_all)
        if progress >= 1:
            status = "\n"
            # status = "\nDone.\n"
        else:
            status = ""
        block = int(round(barLength*progress))
        text = "\r{0} : [{1}] {2}% {3} {4} {5} {6}".format("%10s" % (self.name+"-"+str(self.id)), \
                                                        "#"*block + "-"*(barLength-block), int(progress*100), \
                                                        iter_now, "%.6f"%emod, "%.6f"%esup, status)
        sys.stdout.write(text)
        sys.stdout.flush()

    def run(self, datapack):
        # re-write this function
        return datapack