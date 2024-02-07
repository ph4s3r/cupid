

import h5py
from pathlib import Path
from pprint import pprint

h5file = Path("/mnt/bigdata/placenta/h5-train") / "20230817_135246_a.h5path"

h5file = h5py.File(h5file, 'r')


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