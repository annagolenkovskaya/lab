from analysis.real_data_slices import read_data, trim_myogram
from matplotlib import pylab as plt
from analysis.functions import read_neuron_data
import h5py as hdf5
import numpy as np
path = '/home/anna/Downloads/4 ATP 100mkM (1).mat'
folder = "/".join(path.split("/")[:-1])
# raw_data = read_neuron_data(path)
# print(raw_data)
# raw_data = read_data(path)
# mat_data = trim_myogram(raw_data, folder)
data = []
# data.append([d for d in mat_data[0]])
plt.plot(data)
# plt.show()
# filepath = '/Users/sulgod/Desktop/100mkM.mat'
arrays = {}
f = hdf5.File(path)
for k, v in f.items():
    arrays[k] = np.array(v)
plt.plot(arrays['data'])
plt.show()
print(arrays.keys())