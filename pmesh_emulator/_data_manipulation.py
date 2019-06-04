"""
This file is for reading in the simulation data and saving it into a convenient format. It should not be used by the user. If you are reading this, then disregard. It will do nothing unless you are running locally (Author is Tom McClintock) on my laptop.
"""

import numpy as np

Nboxes = 122
Nsnaps = 30

sf = np.linspace(0.02, 1.0, Nsnaps)

scale_factors = [0.0200, 0.0538, 0.0876, 0.1214, 0.1552]

#Input file path
inpath = "pmesh_test/Box_{box:03d}/powerspec-debug_{scale_factor:0.4f}.txt"

k, _, _ = np.loadtxt(inpath.format(box=0, scale_factor=0.02), unpack=True)

#NOTE! the first k value is 0, and p = 0. This breaks the emulator,
#so we will chop that point off.

k = k[1:]

np.save("k", k)

Nk = k.size

#The emulator interpolates whole cosmologies at once
data = np.zeros((Nboxes, Nk*Nsnaps))

print("k shape {kshape}, p shape {pshape}".format(kshape=k.shape,
                                                  pshape=data.shape))

for box in range(0, 122):
    for i, s in enumerate(sf):
        offset = i*Nk
        _, p, _ = np.loadtxt(inpath.format(box=box, scale_factor=s), \
                             unpack=True)
        #Chop off the k=0 point (the first point)
        p = p[1:]
        data[box, offset: offset+Nk] = p
    print("Done with Box_{box:03d}".format(box=box))

np.save("pkz_data_Nsim_x_NkNz", data)
