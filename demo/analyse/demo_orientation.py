from __future__ import print_function, division, absolute_import

import numpy as np
import h5py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
from spipy.analyse import orientation

if __name__ == "__main__":
    ori_data = 'orientations_100.bin'
    print("\n**1** test orientation.draw_ori_Df ... (actually this function calls another two functions: proc_Hammer and draw_hammer)")

    orientation.draw_ori_Df(ori_bin=ori_data, q_level=10)

    print("\n**2** test orientation.Sphere_randp ...")

    print("\n(1) uniform distribution")
    xyz, theta_phi = orientation.Sphere_randp(algo='uniform-2', radius=50, num=1000)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=5, marker='o')
    plt.show()

    print("\n(2) random distribution")
    xyz, theta_phi = orientation.Sphere_randp(algo='random', radius=50, num=1000)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=5, marker='o')
    plt.show()
