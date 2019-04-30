import numpy as np
import pylab as plt
from analysis.functions import *
from analysis.namespaces import *
# from analysis.bio_data_6runs import bio_several_runs
import matplotlib.patches as mpatches
from analysis.patterns_in_bio_data import bio_data_runs
delta_y_step = 0.3

k_mean = 0
k_x_data = 1
k_y_data = 2


def plot_linear(bio_pack, sim_pack, slices_number, bio_step, sim_step):
	"""
	Args:
		bio_pack (tuple):
			voltages, x-data and y-data
		sim_pack (tuple):
			mean of voltages, x-data and y-data
		slices_number (int):
			number of slices
	"""
	sim_mean_volt = sim_pack[k_mean]
	sim_x = sim_pack[k_x_data]
	sim_y = sim_pack[k_y_data]

	bio_volt = bio_pack[k_mean]
	bio_x = bio_pack[k_x_data]
	bio_y = bio_pack[k_y_data]

	slice_indexes = range(slices_number)

	# plot mean data per slice
	for slice_index in slice_indexes:
		# bio data
		start = int(slice_index * 25 / bio_step)
		end = int((slice_index + 1) * 25 / bio_step)
		sliced_data = bio_volt[start:end]
		offset = slice_index * delta_y_step

		plt.plot([t * bio_step for t in range(len(sliced_data))],
		         [offset + d for d in sliced_data],
		         color='#FF7254')
		# sim data
		start = int(slice_index * 25 / sim_step)
		end = int((slice_index + 1) * 25 / sim_step)
		sliced_data = sim_mean_volt[start:end]

		plt.plot([t * sim_step for t in range(len(sliced_data))],
		         [offset + d for d in sliced_data],
		         color='#54CBFF')

	# BIO data processing
	x = np.array(bio_x)
	y = np.array(bio_y)

	xfit, yfit = calc_linear(x, y)
	num_dots = int(len(x) / 5)
	x_per_test = []
	y_per_test = []
	start = 0
	for i in range(5):
		print(num_dots)
		x_per_test.append(x[start:start + num_dots])
		y_per_test.append(y[start:start + num_dots])
		start += num_dots

	tmp_x = list(zip(*x_per_test))
	tmp_y = list(zip(*y_per_test))

	for i in range(slices_number):
		# print(tmp_x[i])
		plt.scatter(tmp_x[i], tmp_y[i], label=i, color='#CA2D0C', alpha=0.3)
	plt.plot(xfit, yfit, color='#CA2D0C', linestyle='--', linewidth=3, label='BIO')
	# plt.legend()
	# SIM data processing
	x = np.array(sim_x)
	y = np.array(sim_y)

	xfit, yfit = calc_linear(x, y)

	plt.scatter(x, y, color='#25C7FF', alpha=0.3)
	plt.plot(xfit, yfit, color='#0C88CA', linestyle='--', linewidth=3, label='NEST')
	print("xfit = ", xfit)
	print("yfit = ", yfit)
	# plot properties
	plt.xlabel("Time, ms", fontsize=14)
	plt.ylabel("Slices", fontsize=14)
	plt.yticks([])
	plt.xlim(0, 25)
	plt.ylim(-1, slices_number * delta_y_step + 1)
	# plt.legend()
	plt.show()


def form_bio_pack(volt_and_stim, slice_numbers, reversed_data=False):
	bio_lat, bio_amp = bio_process(volt_and_stim, slice_numbers, debugging=False, reversed_data=reversed_data)

	voltages = volt_and_stim[0]
	bio_voltages = normalization(voltages, zero_relative=True)

	bio_x = bio_lat
	bio_y = [index * delta_y_step + bio_voltages[int((bio_x[index] + 25 * index) / bio_step)]
	         for index in range(slice_numbers)]

	# form biological data pack
	bio_pack = (bio_voltages, bio_x, bio_y)

	return bio_pack


def form_sim_pack(tests_data, step, reversed_data=False, debugging=False):
	sim_x = []
	sim_y = []

	# collect amplitudes and latencies per test data
	for test_data in tests_data:
		lat, amp = sim_process(test_data, step, debugging=False)
		print(lat)
		# fixme
		lat = [d if d > 0 else 'fuck' for d in lat ]
		sim_x += lat

		test_data = normalization(test_data, zero_relative=True)

		for slice_index in range(len(lat)):
			a = slice_index * delta_y_step + test_data[int(lat[slice_index] / step + slice_index * 25 / step)]
			sim_y.append(a)

	# form simulation data pack
	sim_mean_volt = normalization(list(map(lambda voltages: np.mean(voltages), zip(*tests_data))), zero_relative=True)
	print("len(sim_mean_volt) = ", len(sim_mean_volt))
	sim_pack = (sim_mean_volt, sim_x, sim_y)

	return sim_pack


def run():
	# bio_volt_and_stim = read_bio_data('../bio-data/3_1.31 volts-Rat-16_5-09-2017_RMG_9m-min_one_step.txt')
	# print("bio_volt_and_stim = ", bio_volt_and_stim[0])
	# print("bio_volt_and_stim = ", bio_volt_and_stim[1])
	nest_tests = read_nest_data('../../nest-data/sim_extensor_eesF40_i100_s15cms_T.hdf5')
	neuron_tests = read_neuron_data('../../neuron-data/15cm.hdf5')
	# print("neuron_tests = ", neuron_tests)
	# plt.plot(neuron_tests)
	# plt.show()
	#
	# slice_numbers = int(len(neuron_tests[0]) * sim_step // 25)

	# FixMe add new functionality for several bio datas
	# result = bio_several_runs()
	# for i in range(len(result)):
		# print("result = ", result[i])
	# bio_volt = result[0]
	# print("bio_volt_and_stim = ", bio_volt_and_stim)
	# bio_volt_and_stim.append(result[1])
	bio_volt = bio_data_runs()
	print(len(bio_volt), "bio_volt = ", bio_volt)
	# print("bio_volt_and_stim = ", bio_volt_and_stim[0])
	# print("bio_volt_and_stim = ", bio_volt_and_stim[-1])
	print("len(bio_volt) = ", len(bio_volt[0]))
	# bio_volt_and_stim2 = result[2]
	# print("bio_volt_and_stim2 = ", bio_volt_and_stim2)
	# bio_volt_and_stim2.append(result[3])
	bio_pack = form_sim_pack(bio_volt, step=bio_step)
	print("printed bio pack")
	# print("count")
	# bio_pack2 = form_sim_pack(bio_volt_and_stim2, step=0.25)
	# print("new count")
	# bio_pack = form_sim_pack(bio_volt, step=0.25, reversed_data=True)

	# loop of NEST and Neuron data
	for sim_tests in [nest_tests, neuron_tests]:
		sim_pack = form_sim_pack(sim_tests, sim_step, debugging=True)
		print("printed sim pack")
		plot_linear(bio_pack, sim_pack, 12, bio_step, sim_step) # min(len(result[1]), len(result[3]))

	quipazine_patch = mpatches.Patch(color='#FF7254', label='quipazine')
	no_quipazine_patch = mpatches.Patch(color='#54CBFF', label='no quipazine')
	plt.legend()


if __name__ == "__main__":
	run()
