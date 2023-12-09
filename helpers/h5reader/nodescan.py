

import h5py
import matplotlib.pyplot as plt
import numpy as np

folder = 'G:\\echinov3\\h5-bad-annotation\\'

file = "echino41.tif.h5path"

h5file = h5py.File(folder+file, 'r')


def scan_hdf5(f, recursive=True, tab_step=2):
    def scan_node(g, tabs=0):
        elems = []
        for k, v in g.items():
            if isinstance(v, h5py.Dataset):
                elems.append(v.name)
            elif isinstance(v, h5py.Group) and recursive:
                elems.append((v.name, scan_node(v, tabs=tabs + tab_step)))
        return elems

    return scan_node(f)

sanyi = scan_hdf5(h5file, recursive=True, tab_step=2)

print(sanyi)