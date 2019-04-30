import numpy as np
import pylab as plt
from analysis.functions import *
from analysis.namespaces import *
from analysis.bio_data_6runs import bio_several_runs, calc_datas_means


def calc_delta(bio_pack, sim_pack):
	diff_lat = [abs(bio - sim) for bio, sim in zip(bio_pack[0], sim_pack[0])]
	diff_amp = [abs(bio - sim) for bio, sim in zip(bio_pack[1], sim_pack[1])]

	return diff_lat, diff_amp


def draw_lat_amp(data_pack):
	"""
	Function for drawing latencies and amplitudes in one plot
	Args:
		data_pack (tuple):
			data pack of latenccies and amplitudes
	"""
	bar_width = 0.2
	latencies = data_pack[0]
	amplitudes = data_pack[1]

	# create axes
	fig, lat_axes = plt.subplots(1, 1, figsize=(15, 12))
	xticks = range(len(amplitudes))

	lat_plot = lat_axes.bar(xticks, latencies, width=bar_width, color=color_lat, alpha=0.7, zorder=2)
	lat_axes.set_xlabel('Slice')
	lat_axes.set_ylabel("Latency, ms", fontsize=12)

	amp_axes = lat_axes.twinx()
	xticks = [x + bar_width for x in xticks]
	amp_plot = amp_axes.bar(xticks, amplitudes, width=bar_width, color=color_amp, alpha=0.7, zorder=2)
	amp_axes.set_ylabel("Amplitude, mV", fontsize=12)

	# plot text annotation for data
	for index in range(len(amplitudes)):
		amp = round(amplitudes[index], 2)
		lat = round(latencies[index], 2)
		lat_axes.text(index - bar_width / 2, lat + max(latencies) / 50, str(lat))
		amp_axes.text(index + bar_width / 2, amp + max(amplitudes) / 50, str(amp))

	plt.legend((lat_plot, amp_plot), ("Latency", "Amplitude"), loc='best')
	plt.show()


def run():
	plot_delta = False

	bio = read_bio_data('../bio-data/3_1.31 volts-Rat-16_5-09-2017_RMG_9m-min_one_step.txt')
	nest_tests = read_nest_data('../../nest-data/sim_extensor_eesF40_i100_s15cms_T.hdf5')
	neuron_tests = read_neuron_data('../../neuron-data/6ST.hdf5')
	gpu_data = read_nest_data('../../GPU_extensor_eesF40_inh100_s15cms_T.hdf5')
	slice_numbers = int(len(neuron_tests[0]) / 25 * sim_step)
	# bio_data = calc_datas_means()[0]
	# bio_indexes = calc_datas_means()[1]
	# bio_data2 = calc_datas_means()[2]
	# bio_indexes2 = calc_datas_means()[3]

	# bio_means = list(map(lambda voltages: np.mean(voltages), zip(*bio_data)))
	# bio_indexes = bio_several_runs()[1]
	# bio = []
	# bio.append(bio_data)
	# bio.append(bio_indexes)
	# bio2 = []
	# bio2.append(bio_data2)
	# bio2.append(bio_indexes2)
	# print(slice_numbers)
	# collect amplitudes and latencies per test data
	if plot_delta:
		bio_pack = bio_process(bio, slice_numbers)
		# bio_pack2 = bio_process(bio2, 15, reversed_data=True)
		# bio_pack = bio_process(bio, slice_numbers)
		nest_pack = sim_process(list(map(lambda voltages: np.mean(voltages), zip(*nest_tests))), sim_step)
		neuron_pack = sim_process(list(map(lambda voltages: np.mean(voltages), zip(*neuron_tests))), sim_step)
		gpu_pack = sim_process(list(map(lambda voltages: np.mean(voltages), zip(*gpu_data))), gpu_step)

		res_pack = calc_delta(bio_pack, neuron_pack)
		draw_lat_amp(res_pack)

		res_pack = calc_delta(bio_pack, neuron_pack)
		# draw_lat_amp(res_pack)

		res_pack = calc_delta(bio_pack, gpu_pack)
		# draw_lat_amp(res_pack)
	else:
		# bio_pack = bio_process(bio, slice_numbers, reversed_data=True)
		# bio_pack = bio_process(bio, slice_numbers)
		# nest_pack = sim_process(list(map(lambda voltages: np.mean(voltages), zip(*nest_tests))), sim_step)
		neuron_pack = sim_process(list(map(lambda voltages: np.mean(voltages), zip(*neuron_tests))), sim_step,
		                          inhibition_zero=False)

		# draw_lat_amp(bio_pack)
		# draw_lat_amp(nest_pack)
		draw_lat_amp(neuron_pack)


if __name__ == "__main__":
	run()
