import numpy as np
from matplotlib import pylab as plt
from analysis.functions import read_bio_hdf5, read_bio_data, convert_bio_to_hdf5, normalization
from analysis.real_data_slices import read_data, trim_myogram
from matplotlib import pylab as plt
raw_mat_data1 = read_data\
	('../bio-data/No Quipazine-Bipedal_SCI Rats_3 volts_40Hz/1_3 volts_NQBiSCI_Rat-1_12-06-2016_RMG&RTA_one_step.mat')
raw_mat_data2  = read_data\
	('../bio-data/No Quipazine-Bipedal_SCI Rats_3 volts_40Hz/3_3 volts_NQBiSCI_Rat-1_12-06-2016_RMG&RTA_one_step.mat')
raw_mat_data3 = read_data\
	('../bio-data/No Quipazine-Bipedal_SCI Rats_3 volts_40Hz/4_3 volts_NQBiSCI_Rat-1_12-06-2016_RMG&RTA_one_step.mat')
raw_mat_data4 = read_data\
	('../bio-data/No Quipazine-Bipedal_SCI Rats_3 volts_40Hz/5_3 volts_NQBiSCI_Rat-1_12-06-2016_RMG&RTA_one_step.mat')
raw_mat_data5 = read_data\
	('../bio-data/No Quipazine-Bipedal_SCI Rats_3 volts_40Hz/10_3 volts_NQBiSCI_Rat-1_12-06-2016_RMG&RTA_one_step.mat')
raw_mat_data2_1 = read_data\
	('../bio-data/No Quipazine quadrupedal SCI 40Hz 3 volts/1_3 volts_NQQuadSCI_Rat-1_12-06-2016_RMG&RTA_one_step.mat')
raw_mat_data2_2 = read_data\
	('../bio-data/No Quipazine quadrupedal SCI 40Hz 3 volts/2_3 volts_NQQuadSCI_Rat-1_12-06-2016_RMG&RTA_one_step.mat')
raw_mat_data2_3 = read_data\
	('../bio-data/No Quipazine quadrupedal SCI 40Hz 3 volts/5_3 volts_NQQuadSCI_Rat-1_12-06-2016_RMG&RTA_one_step.mat')
raw_mat_data2_4 = read_data\
	('../bio-data/No Quipazine quadrupedal SCI 40Hz 3 volts/6_3 volts_NQQuadSCI_Rat-1_12-06-2016_RMG&RTA_one_step.mat')
raw_mat_data2_5 = read_data\
	('../bio-data/No Quipazine quadrupedal SCI 40Hz 3 volts/10_3 volts_NQQuadSCI_Rat-1_12-06-2016_RMG&RTA_one_step.mat')
# raw_mat_data2_6 = read_data\
# 	('../bio-data/Quipazine-Bipedal_SCI Rats_3 volts_40Hz/7Qbi_3 volts__Rat-1_12-05-2016_RMG&RTA_one_step.mat')
mat_data1 = trim_myogram(raw_mat_data1, '../bio-data/No Quipazine-Bipedal_SCI Rats_3 volts_40Hz')
mat_data2 = trim_myogram(raw_mat_data2, '../bio-data/No Quipazine-Bipedal_SCI Rats_3 volts_40Hz')
mat_data3 = trim_myogram(raw_mat_data3, '../bio-data/No Quipazine-Bipedal_SCI Rats_3 volts_40Hz')
mat_data4 = trim_myogram(raw_mat_data4, '../bio-data/No Quipazine-Bipedal_SCI Rats_3 volts_40Hz')
mat_data5 = trim_myogram(raw_mat_data5, '../bio-data/No Quipazine-Bipedal_SCI Rats_3 volts_40Hz')
mat_data2_1 = trim_myogram(raw_mat_data2_1, '../bio-data/Quipazine-Bipedal_SCI Rats_3 volts_40Hz')
mat_data2_2 = trim_myogram(raw_mat_data2_2, '../bio-data/Quipazine-Bipedal_SCI Rats_3 volts_40Hz')
mat_data2_3 = trim_myogram(raw_mat_data2_3, '../bio-data/Quipazine-Bipedal_SCI Rats_3 volts_40Hz')
mat_data2_4 = trim_myogram(raw_mat_data2_4, '../bio-data/Quipazine-Bipedal_SCI Rats_3 volts_40Hz')
mat_data2_5 = trim_myogram(raw_mat_data2_5, '../bio-data/Quipazine-Bipedal_SCI Rats_3 volts_40Hz')
# mat_data2_6 = slice_myogram(raw_mat_data2_6, '../bio-data/Quipazine-Bipedal_SCI Rats_3 volts_40Hz')
print(mat_data1[1])
# raise Exception
datas = []
datas2 = []
# print("len(mat_data_control_15cms_1[0]) = ", len(mat_data_control_15cms_1[0]))
datas.append(mat_data1[0][600:1800])
datas.append(mat_data2[0][900:2100])
datas.append(mat_data3[0][1000:2200])
# datas.append(mat_data4[0])
# datas.append(mat_data5[0])
# print(len(datas), len(datas[0]))
datas2.append(mat_data2_1[0][0:1200])
datas2.append(mat_data2_2[0][0:1200])
datas2.append(mat_data2_3[0][0:1200])
datas2.append(mat_data2_4[0][0:1200])
datas2.append(mat_data2_5[0][0:1200])
# datas2.append(mat_data2_6[0])
print(len(datas[0]), len(mat_data1[1]))
# print(len(mat_data2[0]))
print(len(datas[1]), len(mat_data2[1]))
print(len(datas[2]), len(mat_data3[1]))
# print(len(datas[3]), len(mat_data4[1]))
# print(len(datas[4]), len(mat_data5[1]))
print("-----")
print(len(datas2[0]), len(mat_data2_1[1]))
print(len(datas2[1]), len(mat_data2_2[1]))
print(len(datas2[2]), len(mat_data2_3[1]))
print(len(datas2[3]), len(mat_data2_4[1]))
print(len(datas2[4]), len(mat_data2_5[1]))
# print(len(datas2[5]), len(mat_data2_6[1]))

indexes = mat_data1[1][:25]
# del indexes[-1]
indexes2 = mat_data2_1[1][:20]
indexes2.append(600)
indexes2.append(1000)
indexes2.append(1600)
# indexes2.append(500)
indexes2 = sorted(indexes2)
"""print("indexes = ", indexes)
print("indexes2 = ", indexes2)"""
# for i in range(12):
	# d = read_bio_hdf5('bio_new_{}.hdf5'.format(i + 1))
	# datas.append(normalization(d[0], zero_relative=True))  # 15cms
	# datas.append(d[0])  # 15cms
	# raise Exception
	# indexes.append(d[1])  # 15cms
# for s in range(len(datas)):
	# datas[s] = datas[s][97:1297]
	# print(len(datas[s]), datas[s])
# plt.plot(datas[1])
# for sl in indexes:
# 	plt.axvline(x=sl, linestyle='--', color='gray')
# plt.show()
datas_means = list(map(lambda elements: np.mean(elements), zip(*datas)))  #
print("len(datas_means) = ", len(datas_means), datas_means)
datas_means2 = list(map(lambda elements: np.mean(elements), zip(*datas2)))
# for i in datas:
	# plt.plot(i)
	# plt.plot(datas_means, color='k')

# plt.show()
# for i in datas2:
	# plt.plot(i)
	# plt.plot(datas_means2, color='k')
# plt.show()


def calc_datas_means():
	return datas_means, indexes, datas_means2, indexes2


# plt.plot(datas_means)
# for sl in indexes:
	# plt.axvline(x=sl, linestyle='--', color='gray')
# plt.show()
# plt.plot(datas_means2)
# for sl in indexes2:
	# plt.axvline(x=sl, linestyle='--', color='gray')
# plt.show()
ticks = []
labels = []
for i in range(0, 1201, 100):
	ticks.append(i)
	labels.append(i * 0.25)
	# plt.axvline(x=i, linestyle='--', color='gray')
# plt.xlim(0, 100)
# plt.xticks(ticks, labels)
# for sli in indexes:
	# plt.axvline(x=sli, color='gray', linestyle='--')
# plt.show()# print(datas_means)
# for i in indexes:
	# print(len(i), i)
offset = 0
sliced_mean_data = []
for s in range(12):
	sliced_mean_data_tmp = []
	for i in range(offset, offset + 100):
		# print("i = ", i)
		sliced_mean_data_tmp.append(datas_means[i])
	sliced_mean_data.append(sliced_mean_data_tmp)
	offset += 100
# print(len(sliced_mean_data), len(sliced_mean_data[0]))
# for i in range(12):
	# plt.plot(sliced_mean_data[i])
	# plt.show()
yticks = []
for index, data in enumerate(sliced_mean_data):
	# print("index = ", index)
	# print("data = ", len(data), data)
	offset = index * 2
	yticks.append(data[0] + offset)
	# plt.plot([dat + offset for dat in data], linewidth=0.7)
# plt.yticks(yticks, range(1, len(sliced_mean_data) + 1))
ticks = []
labels = []
for i in range(0, 101, 20):
	ticks.append(i)
	labels.append(i * 0.25)
	# plt.axvline(x=i, linestyle='--', color='gray')
# plt.xlim(0, 100)
# plt.xticks(ticks, labels)
# plt.show()
# print(d1)
# print(len(data_slices))
# yticks = []
# for index, sl in enumerate(range(len(data_slices))):
# 	yticks.append(index * 2)
# 	offset = index * 2
# 	plt.plot([data + offset for data in data_slices[sl]], linewidth = 0.7)
# ticks = []
# labels = []
# for i in range(0, len(data_slices[0]) + 1, 20):
# 	ticks.append(i)
# 	labels.append(i * 0.25)
# 	plt.axvline(x=i, linestyle='--', color='gray')
# print(ticks)
# print(labels)
# plt.xticks(ticks, labels)
# plt.yticks(yticks, range(1, len(data_slices) + 1))
# plt.xlim(0, 100)
# plt.show()
# plt.plot(datas)
# plt.show()


def bio_several_runs():
	print("indexes = ", indexes)
	print("indexes2 = ", indexes2)
	return datas, indexes, datas2, indexes2

# raise Exception
# datas = bio_several_runs()[0]
# indexes = bio_several_runs()[1]

# slice_numbers = len(indexes) - 1

# datas_means = list(map(lambda voltages: np.mean(voltages), zip(*datas)))

# datas_means_per_slice = [[] for _ in range(slice_numbers)]
# for index, data in enumerate(datas_means):
# 	datas_means_per_slice[int(index * 0.25) // 25].append(data)


# global_data_per_test = []
# for data_per_test in datas:
# 	create N (slice numbers) empty lists for current test
	# data_per_slice = [[] for _ in range(slice_numbers)]
	# for index, data in enumerate(data_per_test):
	# 	data_per_slice[int(index * 0.25) // 25].append(data)
	# global_data_per_test.append(data_per_slice)

# tests_per_slice = list(zip(*global_data_per_test))

# for slice_index, slice_data in enumerate(tests_per_slice):
	# print("Slice #{} includes {} tests".format(slice_index, len(slice_data)))
	# for test_data in slice_data:
		# print("\t", len(test_data), test_data)


# yticks = []
# for slice_index, test_per_slice in enumerate(tests_per_slice):
# 	offset = slice_index * 0.5
# 	yticks.append(datas_means_per_slice[slice_index][0] + offset)
# 	times = [time * 0.25 for time in range(len(test_per_slice[0]))]
# 	means = [data + offset for data in datas_means_per_slice[slice_index]]
# 	minimal_per_step = [min(a) for a in zip(*test_per_slice)]
# 	maximal_per_step = [max(a) for a in zip(*test_per_slice)]

	# for maxi, meani, mini in zip(maximal_per_step, means, minimal_per_step):
	# 	if meani > maxi + offset or meani < mini + offset:
	# 		raise Exception("Mean value ({}) is bigger than max ({}) or fewer than min ()".format(slice_index,  maxi,
	# 		                                                                                      meani, mini))
	# plt.plot(times, means, linewidth=0.7)
	# plt.fill_between(times,
	#                  [mini + offset for mini in minimal_per_step],
	#                  [maxi + offset for maxi in maximal_per_step],
	#                  alpha=0.35)

# ticks = []
# labels = []
# for i in range(0, 26, 5):
# 	ticks.append(i)
# 	labels.append(i * 0.25)
# 	plt.axvline(x=i, linestyle='--', color='gray')
# plt.yticks(yticks, range(1, slice_numbers + 1))
# plt.xlim(0, 25)
# plt.show()