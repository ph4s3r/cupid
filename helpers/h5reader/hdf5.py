
import h5py
import matplotlib.pyplot as plt
import numpy as np

folder = 'G:\\echinov3\\h5-bad-annotation\\'

file = "echino41.tif.h5path"

h5file = h5py.File(folder+file, 'r')

item = 0

def convert_int(s):
    try:
        i = int(s)
    except ValueError:
        i = 0
    return i

xcoords = []
ycoords = []

for i in h5file.require_group('tiles').keys():
    x, y = str(i)[1:-2].split(",")
    xcoords.append(convert_int(x))
    ycoords.append(convert_int(y))


print(max(xcoords), max(ycoords))