from . import models
from .models import model_merge
import numpy as np
import os
import json

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
msize = comm.Get_size()

class Runner():

    def __init__(self, inputnode = None, outputnode = None, loadfile = None, change_dataset = None):
        if inputnode is not None and outputnode is not None:
            self.__compile(inputnode, outputnode)
        elif os.path.isfile(loadfile):
            self.load_model(loadfile, change_dataset)
        else:
            raise RuntimeError("[Error] Initiation needs some input !")

    def __compile(self, inputnode, outputnode):
        if type(inputnode) != models.phInput:
            raise RuntimeError("[Error] Input node is invalid !")
        if type(outputnode) != models.phOutput:
            raise RuntimeError("[Error] Output node is invalid !")
        self.inputnode = inputnode
        self.outputnode = outputnode
        self.archive = {}
        # check inputnode -> outputnode
        tmp = self.inputnode
        number = 0
        while tmp != self.outputnode:
            # get rid of repeating name
            i = 1
            tmp_name = tmp.name
            while tmp_name in self.archive.keys():
                tmp_name = tmp.name + "_%d" % i
                i += 1
            self.archive[tmp_name] = number
            tmp.rename(tmp_name)
            number += 1
            # get children
            tmp = tmp.children
            if tmp is None:
                raise RuntimeError("[Error] Cannot move from input node to output node !")
        self.archive[self.outputnode.name] = number

    def load_model(self, model_file, change_dataset = None):
        # load model from json file
        # only support 1 input node with 1 output node
        print("\nLoading model from file '%s' ..." % model_file)
        with open(model_file, "r") as fp:
            input_info, info = json.load(fp)
        if "pattern" not in input_info["config_dict"].keys():
            if change_dataset is None or "pattern_path" not in change_dataset.keys():
                raise ValueError("[Error] Your loaded model is a skeleton model, I need 'pattern_path' in 'change_dataset' parameter !")
        if type(change_dataset) == dict:
            for k, v in change_dataset.items():
                input_info["config_dict"][k] = v
        # get input/output node
        inputnode = models.phInput(input_info["config_dict"], input_info["name"])
        # get all other nodes
        outputnode = inputnode
        children = input_info["children"]
        while children is not None:
            ch_info = info[children]
            thisclass = models.get_model_from_classname(ch_info["classname"])
            outputnode = thisclass(**ch_info["parameters"]).after(outputnode)
            children = ch_info["children"]
        self.__compile(inputnode, outputnode)
        print("Done.")

    def dump_model(self, model_file, skeleton = False):
        if rank != 0:
            return
        # save this model to a json file
        # only support 1 input node with 1 output node, linear structure
        input_info = {"config_dict" : None, "name" : None, "children" : 0}
        info = [None] * (len(self.archive.keys())-1)
        # input node
        input_info["name"] = self.inputnode.name
        input_info["config_dict"] = self.inputnode.config_dict.copy()
        if not skeleton:
            input_info["config_dict"]["pattern"] = np.load(input_info["config_dict"]["pattern_path"]).tolist()
            if input_info["config_dict"]["mask_path"] is not None:
                input_info["config_dict"]["mask"] = np.load(input_info["config_dict"]["mask_path"]).tolist()
            if input_info["config_dict"]["initial_model"] is not None:
                input_info["config_dict"]["initial"] = np.load(input_info["config_dict"]["initial_model"]).tolist()
        input_info["config_dict"]["pattern_path"] = None
        input_info["config_dict"]["mask_path"] = None
        input_info["config_dict"]["initial_model"] = None
        # other nodes
        tmp = self.inputnode
        while True:
            tmp = tmp.children
            # if there are multiple inputnodes, the "index" should be changed
            index = self.archive[tmp.name] - 1 # - len(input_info)
            this_info = {"parameters" : tmp.config_bk, \
                        "classname" : tmp.__class__.__name__, \
                        "children" : None}
            tmp_ch = tmp.children
            if tmp_ch is None:
                info[index] = this_info
                break
            else:
                this_info["children"] = self.archive[tmp_ch.name] - 1 # - len(input_info)
                info[index] = this_info
        with open(model_file, "w") as fp:
            json.dump([input_info, info], fp)
        print("\nDump model to %s." % model_file)

    def run(self, repeat = 1):
        datapack = None

        # repeats
        for j in range(repeat):
            if rank == 0 : print("\n >>> Rank 0 phasing repeat No.%d" % (j+1))
            this_node = self.inputnode
            # for single child node
            while this_node is not None:
                datapack = this_node.run(datapack)
                this_node = this_node.children
        # merge this rank
        out = self.outputnode.merge()

        if msize == 1:
            out["diffraction_amp"] = datapack.copy_diff_amp()
            print("Finished.")
            return out

        # mpi reduce
        comm.Barrier()
        out_all = comm.gather(out, root = 0)

        if rank == 0:
            sample_ret, _ = model_merge.merge_sols(np.array([tmp["sample_ret"] for tmp in out_all]))
            support, _ = model_merge.merge_sols(np.array([tmp["support"] for tmp in out_all]), True)
            PRTF = np.abs(np.mean(np.array([tmp["PRTF"] for tmp in out_all]), axis=0))
            background = np.mean(np.array([tmp["background"] for tmp in out_all]), axis=0)
            eMod = np.mean(np.array([tmp["eMod"] for tmp in out_all]), axis=0)
            eCon = np.mean(np.array([tmp["eCon"] for tmp in out_all]), axis=0)
            out["sample_ret"] = sample_ret
            out["support"] = support
            out["PRTF"] = PRTF
            out["background"] = background
            out["eMod"] = eMod
            out["eCon"] = eCon
            out["diffraction_amp"] = datapack.copy_diff_amp()
            print("Finished.")
            return out
        else:
            return None

    def save_h5(self, out, save_file):
        import h5py
        fp = h5py.File(save_file, "w")
        fp.create_dataset("sample_retrieved", data=out["sample_ret"], chunks=True, compression="gzip")
        fp.create_dataset("support_retrieved", data=out["support"], chunks=True, compression="gzip")
        fp.create_dataset("PRTF", data=out["PRTF"], chunks=True, compression="gzip")
        fp.create_dataset("background", data=out["background"], chunks=True, compression="gzip")
        fp.create_dataset("modulus_error", data=out["eMod"], chunks=True)
        fp.create_dataset("convergence_error", data=out["eCon"], chunks=True)
        fp.create_dataset("diffraction_amplitute", data=out["diffraction_amp"], chunks=True, compression="gzip")
        fp.close()

    def plot_result(self, out):
        import matplotlib.pyplot as plt
        from spipy.image import radp

        prtf = np.abs(np.fft.fftshift(out['PRTF']))
        size = prtf.shape
        prtf_rav = radp.radial_profile(prtf,np.array(size)//2)
        sr = np.abs(np.fft.fftshift(out['sample_ret']))
        dr = np.abs(np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(sr))))
        d = np.abs(np.fft.fftshift(out['diffraction_amp']))
        eCon = out['eCon']
        eMod = out['eMod']
        plt.figure(figsize=(20,10))

        plt.subplot(2,3,1)
        if len(sr.shape) == 2:
            plt.imshow(np.log(1+sr))
        else:
            plt.imshow(np.log(1+sr[:,size[1]//2,:]))
        plt.title('retrieved (real space)')

        plt.subplot(2,3,2)
        if len(sr.shape) == 2:
            plt.imshow(np.log(1+dr))
        else:
            plt.imshow(np.log(1+dr[:,size[1]//2,:]))
        plt.title('retrieved (reciprocal space)')

        plt.subplot(2,3,3)
        if len(sr.shape) == 2:
            plt.imshow(np.log(1+d))
        else:
            plt.imshow(np.log(1+d[:,size[1]//2,:]))
        plt.title('input (reciprocal space)')

        ax1 = plt.subplot(2,3,4)
        plt.plot(eCon,'-k')
        ax1.set_yscale('log')
        plt.xlabel('steps')
        plt.title('convergence')

        ax2 = plt.subplot(2,3,5)
        plt.plot(eMod,'-r')
        ax2.set_yscale('log')
        plt.xlabel('steps')
        plt.title('modulus error')

        plt.subplot(2,3,6)
        qlen = int(np.floor(len(prtf_rav)/np.sqrt(len(sr.shape))))
        qinfo = np.arange(qlen)
        plt.plot(qinfo[:qlen],prtf_rav[:qlen,1],'-k')
        plt.xlabel('q')
        plt.plot(qinfo[:qlen],np.zeros(qlen)+1/np.e,'r--')
        plt.title('PRTF radial average')

        plt.show()