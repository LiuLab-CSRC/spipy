from . import models
from .models import model_merge
import numpy as np
import os

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
msize = comm.Get_size()

class Runner():

    def __init__(self, inputnode=None, outputnode=None):
        if inputnode is not None and outputnode is not None:
            self.compile(inputnode, outputnode)

    def compile(self, inputnode, outputnode):
        if type(inputnode) != models.phInput:
            raise RuntimeError("[Error] Input node is invalid !")
        if type(outputnode) != models.phOutput:
            raise RuntimeError("[Error] Output node is invalid !")
        self.input_node = inputnode
        self.outputnode = outputnode
        # check inputnode -> outputnode
        tmp = self.input_node
        while tmp != self.outputnode:
            tmp = tmp.children
            if tmp is None:
                raise RuntimeError("[Error] Cannot move from input node to output node !")

    def run(self, repeat = 1):
        datapack = None

        # repeats
        for j in range(repeat):
            if rank == 0 : print("\n >>> Rank 0 phasing repeat No.%d" % (j+1))
            this_node = self.input_node
            # for single child node
            while this_node is not None:
                datapack = this_node.run(datapack)
                this_node = this_node.children
        # merge this rank
        out = self.outputnode.merge()

        if msize == 1:
            out["diffraction_amp"] = datapack.copy_diff_amp()
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
            return out
        else:
            return None

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