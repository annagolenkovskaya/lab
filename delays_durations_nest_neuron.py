from analysis.dispersion import read_NEST_data
import numpy as np
import h5py as hdf5
from matplotlib import pylab as plt
from analysis.max_min_values_neuron_nest import calc_max_min
from analysis.peaks_of_real_data_without_EES import delays_nest, delays_neuron, calc_durations_neuron, \
    calc_durations_nest#remove_ees_from_min_max
from mpl_toolkits.mplot3d import Axes3D
neuron_dict = {}


def read_NEURON_data(path):
    with hdf5.File(path, 'r') as f:
        for test_name, test_values in f.items():
            # raw_data = [volt * 10 ** 10 for volt in test_values]
            neuron_dict[test_name] = test_values[:]
    return neuron_dict


def find_mins(array, matching_criteria):
    min_elems = []
    indexes = []
    for index_elem in range(1, len(array) - 1):
        # if symbol == '<':
        if (array[index_elem - 1] > array[index_elem] <= array[index_elem + 1]) and array[index_elem] < \
                matching_criteria:
            min_elems.append(array[index_elem])
            indexes.append(index_elem)
        # if symbol == '>':
        #     if (array[index_elem - 1] > array[index_elem] <= array[index_elem + 1]) and array[index_elem] > matching_criteria:
        #         min_elems.append(array[index_elem])
        #         indexes.append(index_elem)
    return min_elems, indexes


nest_dict = read_NEST_data\
    ('/home/anna/Desktop/lab/nest-data/6cms/40 Hz/sim_healthy_nest_extensor_eesF40_i100_s6cms_T.hdf5')
neuron_dict = read_NEURON_data\
    ('/home/anna/PycharmProjects/LAB/neuron-data/res3010/sim_healthy_neuron_extensor_eesF40_i100_s15cms_T.hdf5')
nest_list = []
neuron_list = []
nest_list = list(nest_dict.values())
neuron_list = list(neuron_dict.values())
# print("len(nest_list) = ", len(nest_list))
# print("len(nest_list) = ", len(nest_list[0]))
# print("len(neuron_list) = ", len(neuron_list))
# print("len(neuron_list) = ", len(neuron_list[0]))
nest_means = list(map(lambda x: np.mean(x), zip(*nest_list)))
# print("len(nest_means) = ", len(nest_means))
neuron_means = list(map(lambda x: np.mean(x), zip(*neuron_list)))
# print("len(neuron_means) = ", len(neuron_means))
slices_begin_time_nest_from_ees = find_mins(nest_means, 12)[1]
slices_begin_time_neuron_from_ees = find_mins(neuron_means, -14 * 10 ** (-10))[1]   # -14
# print("slices_begin_time_neuron = ", slices_begin_time_neuron)
step_nest = slices_begin_time_nest_from_ees[1] - slices_begin_time_nest_from_ees[0]
step_neuron = slices_begin_time_neuron_from_ees[1] - slices_begin_time_neuron_from_ees[0]
slices_begin_time_nest = []
slices_begin_time_neuron = []
offset = 0
for i in range(len(slices_begin_time_nest_from_ees)):
    slices_begin_time_nest.append(offset)
    offset += step_nest
offset = 0
for i in range(len(slices_begin_time_neuron_from_ees)):
    slices_begin_time_neuron.append(offset)
    offset += step_neuron
# print("slices_begin_time_neuron = ", slices_begin_time_neuron)
# slices_begin_time_neuron.append(offset)# slices_begin_time_nest.append(slices_begin_time_nest[-1] + step_nest)
nestFD = 0.1
neuronFD = 0.025
# slices_begin_time_nest = [int(t * nestFD) for t in slices_begin_time_nest]
# print("slices_begin_time_nest = ", slices_begin_time_nest)
data_nest = calc_max_min(slices_begin_time_nest, nest_means, nestFD)
data_neuron = calc_max_min(slices_begin_time_neuron, neuron_means, neuronFD)
# print("slices_max_time = ", data_neuron[0])
# print("slices_max_value = ", data_neuron[1])
# print("slices_min_time = ", data_neuron[2])
# print("slices_min_value = ", data_neuron[3])
#
# # data_nest_with_deleted_ees = remove_ees_from_min_max(data_nest[0], data_nest[1], data_nest[2], data_nest[3])
# # data_neuron_with_deleted_ees = remove_ees_from_min_max(data_neuron[0], data_neuron[1], data_neuron[2], data_neuron[3])
# print("data_neuron_with_deleted_ees[0] = ", data_neuron[0])
# print("data_neuron_with_deleted_ees[1] = ", data_neuron[1])
# print("data_neuron_with_deleted_ees[2] = ", data_neuron[2])
# print("data_neuron_with_deleted_ees[3] = ", data_neuron[3])

# print(data_nest_with_deleted_ees[0])
for i in range(len(data_nest)):
    for key in data_nest[i]:
        if not data_nest[i][key]:
            del data_nest[i][key]
            # print('del')
            break
        # if key == 27:
        #     key = 26
        # if key == '28':
        #     key = '27'
        # if key == '29':
        #     key = '28'
        # if key == '30':
        #     key = '29'
# print("data_nest_with_deleted_ees[0] = ", data_nest_with_deleted_ees[0])
# print("data_nest_with_deleted_ees[1] = ", data_nest_with_deleted_ees[1])
# print("slices_min_value = ", data_nest_with_deleted_ees[2])
# print("slices_min_value = ", data_nest[3])
# plt.plot(nest_means)
# for sl in slices_begin_time_nest:
#     plt.axvline(x=sl, linestyle='--', color='gray')
# plt.show()
max_min_delays_nest = delays_nest(data_nest[0], data_nest[2])
max_min_delays_neuron = delays_neuron(data_neuron[0], data_neuron[2], data_neuron[3])
# print(len(max_min_delays_neuron[0]), "max_delays_neuron = ", max_min_delays_neuron[0])
# print(len(max_min_delays_nest[0]), "max_delays_nest = ", max_min_delays_nest[0])
print(len(max_min_delays_neuron[1]), "min_delays_neuron = ", max_min_delays_neuron[1])
# print(len(max_min_delays_nest[1]), "min_delays_nest = ", max_min_delays_nest[1])
max_min_durations_nest = calc_durations_nest(data_nest[0], data_nest[2])
max_min_durations_neuron = calc_durations_neuron(data_neuron[0], data_neuron[2], data_neuron[3])
# print( len(max_min_durations_neuron[0]), "max_durations_neuron = ", max_min_durations_neuron[0])
# print( len(max_min_durations_nest[0]), "max_durations_nest = ", max_min_durations_nest[0])
print(len(max_min_durations_neuron[1]), "min_durations_neuron = ", max_min_durations_neuron[1])
# print(len(max_min_durations_nest[1]), "min_durations_nest = ", max_min_durations_nest[1])
ticks = []
labels = []
for i in range(0, len(neuron_means), 1000):
    ticks.append(i)
    labels.append(i * 0.025)
# plt.xticks(ticks, labels)
# plt.plot(neuron_means)
# for sl in slices_begin_time_neuron:
#     plt.axvline(x=sl, linestyle='--', color='gray')
# plt.show()
max_delays_delta = []
min_delays_delta = []
max_durations_delta = []
min_durations_delta = []
for i in range(len(max_min_delays_nest[0])):
    max_delays_delta.append(round(max_min_delays_nest[0][i] - max_min_delays_neuron[0][i], 3))
    min_delays_delta.append(round(max_min_delays_nest[1][i] - max_min_delays_neuron[1][i], 3))
    max_durations_delta.append(round(max_min_durations_neuron[0][i] - max_min_durations_nest[0][i], 3))
    min_durations_delta.append(round(max_min_durations_neuron[1][i] - max_min_durations_nest[1][i], 3))
# print("max_delays_delta = ", max_delays_delta)
# print("min_delays_delta = ", min_delays_delta)
# print("max_durations_delta = ", max_durations_delta)
# print("min_durations_delta = ", min_durations_delta)

ticks = []
labels = []
for i in range(0, len(neuron_means), 300):
    ticks.append(i)
    labels.append(i * 0.1)
# plt.xticks(ticks, labels)
max_delays_nest = max_min_delays_nest[0]
max_durations_nest = max_min_durations_nest[0]
min_delays_nest = max_min_delays_nest[1]
min_durations_nest = max_min_durations_nest[1]
max_delays_neuron = max_min_delays_neuron[0]
max_durations_neuron = max_min_durations_neuron[0]
min_delays_neuron = max_min_delays_neuron[1]
min_durations_neuron = max_min_durations_neuron[1]
time = []
for i in range(len(max_delays_nest)):
    time.append(i)
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(time, min_delays_neuron, min_durations_neuron)
ax.plot(time, min_delays_neuron, min_durations_neuron, '.', lw=0.5, color='r', markersize=5)
ax.set_xlabel("Slice number")
ax.set_ylabel("Delays ms")
ax.set_zlabel("Durations ms")
ax.set_title("Slice - Delay - Duration")
plt.show()