

import h5py
import matplotlib.pyplot as plt
import numpy as np
from pprint import pprint

folder = 'G:\\placenta\\h5\\'

file = "20230810_140324.tif.h5path"

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



print(len(sanyi[0]))
pprint(sanyi)
pass