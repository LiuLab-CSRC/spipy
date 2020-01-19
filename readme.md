# Spipy v3.1

### Python toolkits for XFEL single particle imaging (SPI) analysis and reconstruction, compatible with Python3.x

#### Requirement
1. Anaconda with conda version > 4.4
2. GSL/MPI (optional)

#### Content
1. Program -> this repository
2. Wiki -> <https://estonshi.github.io/spipy-wiki/>
3. GUI -> [Github repo](https://github.com/estonshi/spipy_gui/tree/v1.1)

#### Older version
1. version 2.1 & GUI, for Python2.7 -> [Link](https://github.com/LiuLab-CSRC/spipy/tree/examples)

#### Issues
1. It should be noted that for newly built OpenMPI (>3.0.0), same-node shared memory communication support is set to be 'vader' for default and the old 'sm' method is removed. However, the new method make programs hang when message size is large. You can find more details on [Official-Github-Issues](https://github.com/open-mpi/ompi/issues/6568). In spipy, phase.phase3d module is affected by this bug and cannot stop correctly when more than 1 processes are started. To fix it, 'mpiexec' is forced to use '**--mca btl self,tcp**' option, which means using TCP/IP interface to communicate. This is a low-efficiency choice, but also the only choice unfortunately. We will keep track with this problem and update to the newest version of mpi as soon as possible.

> author email : shiyc12 *AT* csrc.ac.cn