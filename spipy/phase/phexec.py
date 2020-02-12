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

    def __init__(self, inputnodes = None, outputnode = None, loadfile = None, reload_dataset = None):
        if inputnodes is not None and outputnode is not None:
            self.__compile(inputnodes, outputnode)
        elif os.path.isfile(loadfile):
            self.load_model(loadfile, reload_dataset)
        else:
            raise RuntimeError("[Error] Initiation needs some input !")

    def __compile(self, inputnodes, outputnode):
        if type(inputnodes) != list:
            raise RuntimeError("[Error] 'inputnodes' should be a list !")
        else:
            for tmp in inputnodes:
                if type(tmp) != models.phInput:
                    raise RuntimeError("[Error] Items in 'inputnodes' should be phInput object !")
        if type(outputnode) != models.phOutput:
            raise RuntimeError("[Error] 'outputnode' should be phOutput object !")
        self.inputnodes = inputnodes
        self.outputnode = outputnode
        self.node_reg = {}
        # multi-input case
        # check inputnodes -> outputnode
        # judge loop
        stack = []
        loop_path = []
        father = {}
        one_path_end = False
        for tmp in self.inputnodes:
            stack.append(tmp)
            self.node_reg[tmp.id] = 1
        while len(stack) > 0:
            tmp = stack.pop(-1)
            # update loop path
            if one_path_end:
                if tmp.id in father.keys():
                    tmp_father = father[tmp.id]
                    if tmp_father.id in loop_path:
                        inx = loop_path.index(tmp_father.id)
                        loop_path = loop_path[:inx+1]
                    else:
                        loop_path.clear()
                else:
                    loop_path.clear()
                one_path_end = False
            # judge loop
            if tmp.id in loop_path:
                raise RuntimeError("[Error] Loop detected in your model !")
            else:
                loop_path.append(tmp.id)
            # get children
            if not tmp.has_children() and tmp != self.outputnode:
                raise RuntimeError("[Error] Cannot move from input node to output node !")
            else:
                # push children into stack
                for child in tmp.children:
                    stack.append(child)
                    father[child.id] = tmp
                    self.node_reg[child.id] = 1
                # set flag when meet outputnode
                if tmp == self.outputnode:
                    one_path_end = True
                else:
                    one_path_end = False


    def load_model(self, model_file, reload_dataset = None, read_data = True):
        '''
        load model from json file
        support multiple input node with 1 output node
        reload_dataset = {id : {"pattern_path" : xxx, ...}, ...}
        '''
        print("\nLoading model from file '%s' ..." % model_file)
        with open(model_file, "r") as fp:
            skeleton, input_node_id, model_set = json.load(fp)
        if skeleton:
            if reload_dataset is None:
                raise ValueError("[Error] Your loaded model is a skeleton model, 'change_dataset' is required !")
        if reload_dataset is not None:
            for id, dataset in reload_dataset.items():
                if id not in input_node_id:
                    raise RuntimeError("[Error] The 'id' in reload_dataset should be an inputnode id !")
                if "pattern_path" not in dataset.keys():
                    raise RuntimeError("[Error] The data in reload_dataset must contain 'pattern_path' item !")
                for k, v in dataset.items():
                    model_set[str(id)]["parameters"]["config_dict"][k] = v
                    if skeleton is False and read_data is True:
                        kk = k.split("_")[0]
                        model_set[str(id)]["parameters"]["data_reload"].pop(kk)
        # get input/output node
        register = {}   # new node need to register
        id_map = {}     # {id_in_new_model : id_in_old_model, ...}
        inputnodes = []
        for old_id in input_node_id:
            if skeleton is False and read_data is False:
                model_set[str(old_id)]["parameters"]["data_reload"].clear()
            new_node = models.phInput(**model_set[str(old_id)]["parameters"])
            inputnodes.append(new_node)
            register[old_id] = new_node
            id_map[new_node.id] = old_id
        # get all other nodes
        stack = []
        outputnode = None
        for tmp in inputnodes:
            stack.append(tmp)
        while len(stack) > 0:
            this_node = stack.pop(-1)
            this_info = model_set[str(id_map[this_node.id])]
            if this_info["classname"] == models.phOutput.__name__:
                outputnode = this_node
            for child_id in this_info["children"]:
                if child_id in register.keys():
                    register[child_id] = register[child_id].after(this_node)
                    continue
                ch_info = model_set[str(child_id)]
                thisclass = models.get_model_from_classname(ch_info["classname"])
                ch_node = thisclass(**ch_info["parameters"]).after(this_node)
                stack.append(ch_node)
                register[child_id] = ch_node
                id_map[ch_node.id] = child_id
        # compile
        self.__compile(inputnodes, outputnode)
        print("Done.")


    def dump_model(self, model_file, skeleton = False):
        if rank != 0:
            return
        # save this model to a json file
        # only support multiple input node with 1 output node, loop is not allowed
        model_set = {}
        input_node_id = []
        # all nodes
        stack = []
        register = self.node_reg.copy()
        # start iteration
        for tmp in self.inputnodes:
            stack.append(tmp)
        while len(stack) > 0:
            tmp = stack.pop(-1)
            if tmp.id in register.keys():
                this_info = {"parameters" : tmp.config_bk, \
                            "classname" : tmp.__class__.__name__, \
                            "children" : []}
                # push children
                for child in tmp.children:
                    stack.append(child)
                    this_info["children"].append(child.id)
                # if this is a input node
                if tmp in self.inputnodes:
                    input_node_id.append(tmp.id)
                    if skeleton is False:
                        this_info["parameters"]["data_reload"] = \
                        {
                            "pattern" : np.load(tmp.config_bk["config_dict"]["pattern_path"]).tolist(), \
                            "mask" : None, \
                            "initial" : None \
                        }
                        if tmp.config_bk["config_dict"]["mask_path"] is not None:
                            this_info["parameters"]["data_reload"]["mask"] = \
                                np.load(tmp.config_bk["config_dict"]["mask_path"]).astype(int).tolist()
                        if tmp.config_bk["config_dict"]["initial_model"] is not None:
                            this_info["parameters"]["data_reload"]["initial"] = \
                                np.load(tmp.config_bk["config_dict"]["initial_model"]).tolist()
                    # the data paths in this model will not be transferred to the new one
                    this_info["parameters"]["config_dict"]["pattern_path"] = None
                    this_info["parameters"]["config_dict"]["mask_path"] = None
                    this_info["parameters"]["config_dict"]["initial_model"] = None
                # fill in model set
                model_set[tmp.id] = this_info
                # del tmp from register
                register.pop(tmp.id)
        with open(model_file, "w") as fp:
            json.dump([skeleton, input_node_id, model_set], fp)
        print("\nDump model to %s." % model_file)

    def run(self, repeat = 1):
        datapack = None
        stack = []
        datapack_buff = {}
        # repeats
        for j in range(repeat):
            if rank == 0 : print("\n>>> Rank 0 phasing repeat No.%d" % (j+1))
            # for mulit-child node, use a stack
            stack.clear()
            datapack_buff.clear()
            datapack = None
            for tmp in self.inputnodes:
                stack.append(tmp)
            while len(stack) > 0 :
                this_node = stack.pop(-1)
                if this_node.id in datapack_buff.keys():
                    datapack = datapack_buff.pop(this_node.id)
                this_node.set_repeat(j)
                datapack = this_node.run(datapack)
                if datapack is not None:
                    # not a phMerge node
                    children_num = len(this_node.children)
                    for i,tmp in enumerate(this_node.children):
                        stack.append(tmp)
                        if i + 1 < children_num:
                            datapack_buff[tmp.id] = datapack.copy()
        # merge this rank
        out = self.outputnode.merge()

        if msize == 1:
            out["diffraction_amp"] = datapack.copy_diff_amp()
            print(">>> Finished.\n")
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
            print(">>> Finished.\n")
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
        eCon = np.concatenate(out['eCon'])
        eMod = np.concatenate(out['eMod'])
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