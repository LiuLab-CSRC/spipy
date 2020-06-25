from __future__ import print_function, division, absolute_import

import numpy as np
from spipy.analyse import saxs
import matplotlib.pyplot as plt
import h5py

if __name__ == "__main__":
    dataset = h5py.File('../test_pattern.h5', 'r')['patterns'][()]
    mask = h5py.File('../test_pattern.h5', 'r')['mask'][()]

    print("\ncalculate saxs pattern of a dataset ..")
    saxs_pat = saxs.cal_saxs(data=dataset)

    print("\nSearch zero frequency point of saxs pattern ..")
    cen = saxs.friedel_search(pattern=saxs_pat, estimated_center=(64, 64), mask=mask, small_r=10, large_r=40)
    print("Center is : " + str(cen))

    print("\nCalculate averaged intensity radial profile of a dataset ..")
    inten_prof = saxs.inten_profile_vfast(dataset, mask, 581, 7.5, 64, 0.55)
    inten_prof_2 = saxs.inten_profile_vaccurate(dataset, mask, 581, 7.5, 64, 0.55)

    plt.subplot(1,2,1)
    plt.plot(inten_prof[:, 0], np.log(1 + np.abs(inten_prof[:, 1])), 'r-')
    plt.title('vfast')
    plt.xlabel('q')
    plt.ylabel('radial intensity profile')
    plt.subplot(1,2,2)
    plt.plot(inten_prof_2[:, 0], np.log(1 + np.abs(inten_prof_2[:, 1])), 'r-')
    plt.title('vaccurate')
    plt.xlabel('q')
    plt.ylabel('radial intensity profile')
    plt.show()

    print("\nEvaluate particle size using saxs profile...")
    (estimated, radial_intensity) = saxs.particle_size(
        saxs=saxs_pat, estimated_center=cen, exparam='578,7.5,0.55', high_filter_cut=0.3,
        power=0.8, mask=mask)
    print("estimated particle size : " + str(estimated) + " nm")
    plt.plot(radial_intensity[:, 0], radial_intensity[:, 1], 'r-')
    plt.plot(np.zeros(50) + 90, np.linspace(0, 0.2, 50), 'k-')
    plt.show()

    print("\nEvaluate particle sizes using lsq fitting...")
    D = saxs.particle_size_sp(dataset=dataset, exparam=[581,7.5,0.55], fitarea=[10,40], badsearchr=60, method="lsq", mask=mask, center=cen, verbose=True)
    plt.hist(D, bins=20)
    plt.show()
