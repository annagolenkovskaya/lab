import csv
import logging
import numpy as np
import h5py as hdf5
import pylab as plt
from sklearn.decomposition import PCA


logging.basicConfig(format='[%(funcName)s]: %(message)s', level=logging.INFO)
log = logging.getLogger()


def normalization(data, a=0, b=1, save_centering=False):
	"""
	Normalization in [a, b] interval or with saving centering
	x` = (b - a) * (xi - min(x)) / (max(x) - min(x)) + a
	Args:
		data (np.ndarray): data for normalization
		a (float): left interval
		b (float): right interval
		save_centering (bool): if True -- will save data centering and just normalize by lowest data
	Returns:
		np.ndarray: normalized data
	"""
	# checking on errors
	if a >= b:
		raise Exception("Left interval 'a' must be fewer than right interval 'b'")
	if save_centering:
		minimal = abs(min(data))
		return [volt / minimal for volt in data]
	else:
		min_x = min(data)
		max_x = max(data)
		const = (b - a) / (max_x - min_x)

		return [(x - min_x) * const + a for x in data]


<<<<<<< HEAD
def calc_linear(x, y):
	model = LinearRegression(fit_intercept=True)
	model.fit(x[:, np.newaxis], y)
	xfit = np.linspace(0, 25, 10)
	yfit = model.predict(X=xfit[:, np.newaxis])

	return xfit, yfit


def calc_max_min(slices_start_time, test_data, remove_micropeaks=False, stim_corr=None, find_EES=False):
	"""
	Function for finding min/max extrema
	Args:
		slices_start_time (list or range):
			list of slices start times
		test_data (list):
			list of data for processing
		step (float):
			step size of data recording
		remove_micropeaks (optional or bool):
			True - if need to remove micro peaks (<0.02 mV of normalized data)
		stim_corr (list):
			EES stimulation indexes for correction the time of found min/max points (to be relative from EES stim)
	Returns:
		(list): slices_max_time
		(list): slices_max_value
		(list): slices_min_time
		(list): slices_min_value
	"""
	# print("slices_start_time = ", slices_start_time)
	slices_max_time = []
	slices_max_value = []
	slices_min_time = []
	slices_min_value = []

	for slice_index in range(1, len(slices_start_time) + 1):
		tmp_max_time = []
		tmp_min_time = []
		tmp_max_value = []
		tmp_min_value = []

		if stim_corr:
			offset = slices_start_time[slice_index - 1] - stim_corr[slice_index - 1]
		else:
			offset = 0

		start = slices_start_time[slice_index - 1]
		if slice_index == len(slices_start_time):
			end = len(test_data)
		else:
			end = slices_start_time[slice_index]

		sliced_values = test_data[start:end]
		border = len(sliced_values) / 3 if find_EES else len(sliced_values)
		datas_times = range(end - start)
		# compare points
		for i in range(1, len(sliced_values) - 1):
			if sliced_values[i - 1] < sliced_values[i] >= sliced_values[i + 1] and i < border:
				tmp_max_time.append(datas_times[i] + offset)
				tmp_max_value.append(sliced_values[i])
			if sliced_values[i - 1] > sliced_values[i] <= sliced_values[i + 1] and i < border:
				tmp_min_time.append(datas_times[i] + offset)
				tmp_min_value.append(sliced_values[i])
			if not tmp_max_time or not tmp_max_value or not tmp_min_time or not tmp_min_value:
				border += 1

		# append found points per slice to the 'main' lists
		slices_max_time.append(tmp_max_time)
		slices_max_value.append(tmp_max_value)
		slices_min_time.append(tmp_min_time)
		slices_min_value.append(tmp_min_value)

	# FixMe remove this functionality in future
	if remove_micropeaks:
		raise Warning("This functionality is deprecated and will be removed soon")
		# small realization of ommiting data marked as False
		remove_micropeaks_func = lambda datas, booleans: [data for data, boolean in zip(datas, booleans) if boolean]

		diff = 0.02 # the lowest difference between two points value which means micro-changing
		# per slice
		for slice_index in range(len(slices_min_value)):
			max_i = 0
			min_i = 0
			len_max = len(slices_max_time[slice_index])
			len_min = len(slices_min_time[slice_index])
			# init by bool the tmp lists for marking points
			maxes_bool = [True] * len_max
			mins_bool = [True] * len_min
			# just simplification
			maxes_val = slices_max_value[slice_index]
			mins_val = slices_min_value[slice_index]
			maxes_time = slices_max_time[slice_index]
			mins_time = slices_min_time[slice_index]

			while (max_i < len_max - 1) and (min_i < len_min - 1):
				# if points have small differnece mark them as False
				if abs(maxes_val[max_i] - mins_val[min_i]) < diff:
					maxes_bool[max_i] = False
					mins_bool[min_i] = False
				# but if the current points has the 3ms difference with the next point, remark the current as True
				if abs(mins_time[min_i + 1] - mins_time[min_i]) > (1 / sim_step):
					mins_bool[min_i] = True
				if abs(maxes_time[max_i + 1] - maxes_time[max_i]) > (1 / sim_step):
					maxes_bool[max_i] = True
				# change indexes (walking by pair: min-max, max-min, min-max...)
				if max_i == min_i:
					max_i += 1
				else:
					min_i += 1
			# ommit the data marked as False
			slices_max_value[slice_index] = remove_micropeaks_func(maxes_val, maxes_bool)
			slices_max_time[slice_index] = remove_micropeaks_func(maxes_time, maxes_bool)
			slices_min_value[slice_index] = remove_micropeaks_func(mins_val, mins_bool)
			slices_min_time[slice_index] = remove_micropeaks_func(mins_time, mins_bool)

	return slices_max_time, slices_max_value, slices_min_time, slices_min_value


def find_latencies(mins_maxes, step, norm_to_ms=False, reversed_data=False, inhibition_zero=False, first_kink=False):
	"""
	Function for autonomous finding the latencies in slices by bordering and finding minimals
	Args:
		mins_maxes (list of list):
			0 max times by slices
			1 max values by slices
			2 min times by slices
			3 min values by slices
		step (float or int):
			step of data recording (e.g. step=0.025 means 40 recorders in 1 ms)
		norm_to_ms (bool):
			if True -- convert steps to ms, else return steps
	Returns:
		list: includes latencies for each slice
	"""
	latencies = []
	times_latencies = []
	slice_numbers = len(mins_maxes[0])
	slice_indexes = range(slice_numbers)

	slices_index_interval = lambda a, b: slice_indexes[int(slice_numbers / 6 * a):int(slice_numbers / 6 * (b + 1))]
	step_to_ms = lambda current_step: current_step * step
	count = 0
	# find latencies per slice
	for slice_index in slice_indexes:
		additional_border = 0
		slice_times = mins_maxes[0][slice_index]    # was 2
		for time in mins_maxes[2][slice_index]:
			slice_times.append(time)
		slice_values = mins_maxes[1][slice_index]   # was 3
		for value in mins_maxes[3][slice_index]:
			slice_values.append(value)
		slice_times, slice_values = (list(x) for x in zip(*sorted(zip(slice_times, slice_values))))
		# while minimal value isn't found -- find with extended borders [left, right]

		while True:
			if inhibition_zero:
				left = 10 - additional_border   # 11
				right = 25 + additional_border
			else:
				# raise Exception
				if slice_index in slices_index_interval(0, 1): # [0,1]
					if reversed_data:
						left = 15 - additional_border
						right = 24 + additional_border
					else:
						left = 12 - additional_border
						right = 18 + additional_border
				elif slice_index in slices_index_interval(2, 2): # [2]
					if reversed_data:
						left = 13 - additional_border
						right = 21 + additional_border
					else:
						left = 15 - additional_border
						right = 17 + additional_border
				elif slice_index in slices_index_interval(3, 4): # [3, 4]
					if reversed_data:
						left = 10 - additional_border
						right = 17 + additional_border
					else:
						left = 15 - additional_border
						right = 21 + additional_border
				elif slice_index in slices_index_interval(5, 6): # [5, 6]
					if reversed_data:
						left = 11 - additional_border
						right = 16 + additional_border
					else:
						left = 13 - additional_border
						right = 24 + additional_border
				else:
					raise Exception("Error in the slice index catching")

			if left < 0:
				left = 0
			if right > 25:
				right = 25

			found_points = [v for i, v in enumerate(slice_values) if left <= step_to_ms(slice_times[i]) <= right]
			# for f in found_points:
				# if slice_values[slice_values.index(f) > thresholds[count]]:
					# latencies.append(slice_times[slice_values.index(f)])

			# save index of the minimal element in founded points
			if len(found_points):
				# for f in range(len(found_points)):
			# 		if slice_values[slice_values.index(found_points[f])] > thresholds[count]:
						# latencies.append(slice_times[slice_values.index(found_points[f])])
						# count += 1
						# break
				# else:
				# 		f += 1
				# for i in range(len(found_points)):
					# if found_points[i] <= found_points[i + 1]:
						# minimal_val = found_points[i]
						# print("minimal_val = ", minimal_val)
						# break
				minimal_val = found_points[0] if first_kink else found_points[0] # found_points[0]
				index_of_minimal = slice_values.index(minimal_val)
				latencies.append(slice_times[index_of_minimal])
				break

			else:
				additional_border += 1
			if additional_border > 25:
				# FixMe
				latencies.append(-999)
				break
				# FixMe raise Exception("Error, out of borders")
	# checking on errors
	if len(latencies) != slice_numbers:
		raise Exception("Latency list length is not equal to number of slices!")

	if norm_to_ms:
		return [lat * step for lat in latencies]
	return latencies


def find_ees_indexes(stim_indexes, datas, reverse_ees=False):
	"""
	Function for finding the indexes of the EES mono-answer in borders formed by stimulations time
	Args:
		stim_indexes (list):
			indexes of the EES stimulations
		datas (list of list):
			includes min/max times and min/max values
	Returns:
		list: global indexes of the EES mono-answers
	"""
	ees_indexes = []

	if reverse_ees:
		for slice_index in range(len(stim_indexes)):
			max_values = datas[k_max_val][slice_index]
			max_times = datas[k_max_time][slice_index]
			# EES peak is the minimal one
			ees_value_index = max_values.index(max(max_values))
			# calculate the EES answer as the sum of the local time of the found EES peak (of the layer)
			# and global time of stimulation for this layer
			ees_indexes.append(stim_indexes[slice_index] + max_times[ees_value_index])
	else:
		for slice_index in range(len(stim_indexes)):
			min_values = datas[k_min_val][slice_index]
			min_times = datas[k_min_time][slice_index]
			# EES peak is the minimal one
			ees_value_index = min_values.index(min(min_values))
			# calculate the EES answer as the sum of the local time of the found EES peak (of the layer)
			# and global time of stimulation for this layer
			ees_indexes.append(stim_indexes[slice_index] + min_times[ees_value_index])
	return ees_indexes


def calc_amplitudes(datas, latencies, step, ees_end, after_latencies=False):
	"""
	Function for calculating amplitudes
	Args:
		datas (list of list):
			includes min/max time min/max value for each slice
		latencies (list):
			latencies pr slice for calculating only after the first poly-answer
	Returns:
		list: amplitudes per slice
	"""
	amplitudes = []
	slice_numbers = len(datas[0])
	dots_per_slice = 0
	if step == 0.25:
		dots_per_slice = 100
	if step == 0.025:
		dots_per_slice = 1000
	for l in range(len(latencies)):
		latencies[l] /= step
		latencies[l] = int(latencies[l])
	# print("latencies in ampls= ", latencies)
	max_times = datas[0]
	max_values = datas[1]
	min_times = datas[2]
	min_values = datas[3]

	# print("amp latencies = ", latencies)
	# print("amp max_times = ", max_times)
	# print("amp max_values = ", max_values)
	# print("amp min_times = ", min_times)
	# print("amp min_values = ", min_values)
	# print("max_values = (func)", max_values)
	# print("min_times = ", min_times)
	max_times_amp = []
	min_times_amp = []
	max_values_amp = []
	min_values_amp = []

	# print("latencies in amp cycle = ", latencies)
	for i in range(len(latencies)):
		max_times_amp_tmp = []
		for j in range(len(max_times[i])):
			if max_times[i][j] > latencies[i]:
				max_times_amp_tmp.append(max_times[i][j])
		max_times_amp.append(max_times_amp_tmp)
		min_times_amp_tmp = []
		for j in range(len(min_times[i])):
			if min_times[i][j] > latencies[i]:
				min_times_amp_tmp.append(min_times[i][j])
		min_times_amp.append(min_times_amp_tmp)

		max_values_amp.append(max_values[i][len(max_times[i]) - len(max_times_amp[i]):])
		min_values_amp.append(min_values[i][len(min_times[i]) - len(min_times_amp[i]):])

	# print("amp max_times = ", max_times)
	# print("amp max_values = ", max_values)
	# print("amp min_times = ", min_times)
	# print("amp min_values = ", min_values)

	corrected_max_times_amp = []
	corrected_max_values_amp = []

	wrong_sl = []
	wrong_dot = []

	for index_sl, sl in enumerate(max_times_amp):
		corrected_max_times_amp_tmp = []
		for index_dot, dot in enumerate(sl):
			if dot < dots_per_slice:
				corrected_max_times_amp_tmp.append(dot)
			else:
				wrong_sl.append(index_sl)
				wrong_dot.append(index_dot)
		corrected_max_times_amp.append(corrected_max_times_amp_tmp)

	corrected_max_values_amp = max_values_amp
	for i in range(len(wrong_sl) - 1, -1, -1):
		del corrected_max_values_amp[wrong_sl[i]][wrong_dot[i]]

	corrected_min_times_amp = []
	corrected_min_values_amp = []
	wrong_sl = []
	wrong_dot = []

	for index_sl, sl in enumerate(min_times_amp):
		corrected_min_times_amp_tmp = []
		for index_dot, dot in enumerate(sl):
			if dot < dots_per_slice:
				corrected_min_times_amp_tmp.append(dot)
			else:
				wrong_sl.append(index_sl)
				wrong_dot.append(index_dot)
		corrected_min_times_amp.append(corrected_min_times_amp_tmp)

	corrected_min_values_amp = min_values_amp
	for i in range(len(wrong_sl) - 1, -1, -1):
		del corrected_min_values_amp[wrong_sl[i]][wrong_dot[i]]

	for sl in range(len(corrected_min_times_amp)):
		for dot in range(1, len(corrected_min_times_amp[sl])):
			if corrected_min_times_amp[sl][dot - 1] > corrected_min_times_amp[sl][dot]:
				corrected_min_times_amp[sl] = corrected_min_times_amp[sl][:dot]
				corrected_min_values_amp[sl] = corrected_min_values_amp[sl][:dot]

	for sl in range(len(corrected_max_times_amp)):
		for dot in range(1, len(corrected_max_times_amp[sl])):
			if corrected_max_times_amp[sl][dot - 1] > corrected_max_times_amp[sl][dot]:
				corrected_max_times_amp[sl] = corrected_max_times_amp[sl][:dot]
				corrected_max_values_amp[sl] = corrected_max_values_amp[sl][:dot]
				break

	peaks_number = []
	for sl in range(len(corrected_min_values_amp)):
		peaks_number.append(len(corrected_min_values_amp[sl]) + len(corrected_max_values_amp[sl]))
	amplitudes = []
	# print("corrected_min_values_amp = ", corrected_min_values_amp)
	# print("corrected_max_values_amp = ", corrected_max_values_amp)

	for sl in range(len(corrected_max_values_amp)):
		# print("sl = ", sl)
		amplitudes_sl = []
		amplitudes_sum = 0
		try:
			for i in range(len(corrected_max_values_amp[sl]) - 1):
				# print("i = ", i)
				amplitudes_sl.append(corrected_max_values_amp[sl][i] - corrected_min_values_amp[sl][i])
				amplitudes_sl.append(corrected_max_values_amp[sl][i + 1] - corrected_min_values_amp[sl][i])
		except IndexError:
			continue

		for amp in amplitudes_sl:
			amplitudes_sum += amp

		amplitudes.append(amplitudes_sum)

	for l in range(len(latencies)):
		latencies[l] *= step

	return amplitudes, peaks_number, corrected_max_times_amp, corrected_max_values_amp, corrected_min_times_amp, \
	       corrected_min_values_amp


def debug(voltages, datas, stim_indexes, ees_indexes, latencies, amplitudes, step):
	"""
	Temporal function for visualization of preprocessed data
	Args:
		voltages (list):
			voltage data
		datas (list of list):
			includes min/max time min/max value for each slice
		stim_indexes (list):
			indexes of EES stimlations
		ees_indexes (list):
			indexes of EES answers (mono-answer)
		latencies (list):
			latencies of the first poly-answers per slice
		amplitudes (list):
			amplitudes per slice
		step (float):
			 step size of the data
	"""
	amplitudes_y = []

	slice_indexes = range(len(ees_indexes))

	show_text = True
	show_amplitudes = True
	show_points = True
	show_axvlines = True

	# the 1st subplot demonstrates a voltage data, ees answers, ees stimulations and found latencies
	ax = plt.subplot(2, 1, 1)
	# plot the voltage data
	norm_voltages = normalization(voltages, zero_relative=True)

	plt.plot([t * step for t in range(len(norm_voltages))], norm_voltages, color='grey', linewidth=1)
	# standartization to the step size
	for slice_index in slice_indexes:
		datas[k_max_time][slice_index] = [d * step for d in datas[0][slice_index]]
	for slice_index in slice_indexes:
		datas[k_min_time][slice_index] = [d * step for d in datas[2][slice_index]]

	stim_indexes = [index * step for index in stim_indexes]
	ees_indexes = [index * step for index in ees_indexes]

	# plot the EES stimulation
	for i in stim_indexes:
		if show_axvlines:
			plt.axvline(x=i, linestyle='--', color='k', alpha=0.35, linewidth=1)
		if show_text:
			plt.annotate("Stim\n{:.2f}".format(i), xy=(i-1.3, 0), textcoords='data', color='k')
	# plot the EES answers
	for index, v in enumerate(ees_indexes):
		if show_axvlines:
			plt.axvline(x=v, color='r', alpha=0.5, linewidth=5)
		if show_text:
			plt.annotate("EES\n{:.2f}".format(v - stim_indexes[index]), xy=(v + 1, -0.2), textcoords='data', color='r')
	# plot the latencies
	for index, lat in enumerate(latencies):
		lat_x = stim_indexes[index] + lat
		if show_axvlines:
			plt.axvline(x=lat_x, color='g', alpha=0.7, linewidth=2)
		if show_text:
			plt.annotate("Lat: {:.2f}".format(lat), xy=(lat_x + 0.2, -0.4), textcoords='data', color='g')
	# plot min/max points for each slice and calculate their amplitudes
	plt.axhline(y=0, color='r', linestyle='--', linewidth=1)
	plt.axhline(y=-1, color='r', linestyle='--', linewidth=1)

	for slice_index in slice_indexes:
		min_times = datas[k_min_time][slice_index]
		min_values = datas[k_min_val][slice_index]
		max_times = datas[k_max_time][slice_index]
		max_values = datas[k_max_val][slice_index]
		amplitudes_y.append(amplitudes[slice_index])
		# plot them
		if show_points:
			plt.plot([t + stim_indexes[slice_index] for t in min_times], min_values, '.', color='b', markersize=5)
			plt.plot([t + stim_indexes[slice_index] for t in max_times], max_values, '.', color='r', markersize=5)
	plt.legend()

	# plot the amplitudes with shared x-axis
	plt.subplot(2, 1, 2, sharex=ax)
	# plot the EES answers
	if show_amplitudes:
		for i in ees_indexes:
			plt.axvline(x=i, color='r')
		# plot amplitudes by the horizontal line
		plt.bar([ees_index + ees_indexes[0] for ees_index in ees_indexes], amplitudes, width=5, color=color_lat,
		        alpha=0.7, zorder=2)
		for slice_index in slice_indexes:
			x = ees_indexes[slice_index] + ees_indexes[0] - 5 / 2
			y = amplitudes[slice_index]
			plt.annotate("{:.2f}".format(y), xy=(x, y + 0.01), textcoords='data')
		plt.ylim(0, 0.8)
	else:
		plt.plot([t * step for t in range(len(voltages))], voltages, color='grey', linewidth=1)
	plt.xlim(0, 150)
	plt.show()
	plt.close()


def __process(latencies, voltages, stim_indexes, step, ees_end, debugging, inhibition_zero=True, reverse_ees=False,
              after_latencies=False, first_kink=False):
	"""
	Unified functionality for finding latencies and amplitudes
	Args:
		voltages (list):
			voltage data
		stim_indexes (list):
			EES stimulations indexes
		step (float):
			step size of data
		debugging (bool):
			If True -- plot all found variables for debugging them
	Returns:
		list: latencies -- latency per slice
		list: amplitudes -- amplitude per slice
	"""
	mins_maxes = calc_max_min(stim_indexes, voltages, find_EES=True)   # check
	ees_indexes = find_ees_indexes(stim_indexes, mins_maxes, reverse_ees=reverse_ees)
	norm_voltages = normalization(voltages, zero_relative=True)
	mins_maxes = calc_max_min(ees_indexes, voltages, stim_corr=stim_indexes)
	# latencies = find_latencies(mins_maxes, step, norm_to_ms=True, reversed_data=reversed_data,
	#                            inhibition_zero=inhibition_zero, first_kink=first_kink) # , thresholds
	amplitudes, peaks_number, max_times, min_times, max_values, min_values = \
		calc_amplitudes(mins_maxes, latencies, step, ees_end, after_latencies)

	# if debugging:
	# 	debug(voltages, mins_maxes, stim_indexes, ees_indexes, latencies, amplitudes, step)
	return amplitudes, peaks_number, max_times, min_times, max_values, min_values


def bio_process(voltages_and_stim, slice_numbers, debugging=False, reversed_data=False, reverse_ees=False):
	"""
	Find latencies in EES mono-answer borders and amplitudes relative from zero
	Args:
		voltages_and_stim (list):
			 voltages data and EES stim indexes
		slice_numbers (int):
			number of slices which we need to use in comparing with simulation data
		debugging (bool):
			If True -- plot all found variables for debugging them
	Returns:
		tuple: bio_lat, bio_amp -- latencies and amplitudes per slice
	"""
	# form EES stimulations indexes (use only from 0 to slice_numbers + 1)
	stim_indexes = voltages_and_stim[k_bio_stim][:slice_numbers + 1]    # +1
	# remove unescesary voltage data by the last EES stimulation index
	voltages = voltages_and_stim[k_bio_volt][:stim_indexes[-1]]
	volts_by_stims = []
	thresholds = []
	offset = 0
	for i in range(int(len(voltages) / 100)):
		volts_by_stims_tmp = []
		for j in range(offset, offset + 100):
			volts_by_stims_tmp.append(voltages[j])
		volts_by_stims.append(volts_by_stims_tmp)
		offset += 100
	for v in volts_by_stims:
		thresholds.append(0.137 * max(v))
	stim_indexes = stim_indexes[:-1]
	# calculate the latencies and amplitudes
	bio_lat, bio_amp = __process(voltages, stim_indexes, bio_step, debugging, reversed_data=reversed_data,
	                             reverse_ees=reverse_ees)

	return bio_lat, bio_amp


def sim_process(latencies, voltages, step, ees_end, debugging=False, inhibition_zero=False, after_latencies=False,
                first_kink=False):
	"""
	Find latencies in EES mono-answer borders and amplitudes relative from zero
	Args:
		voltages (list):
			 voltages data
		debugging (bool):
			If True -- plot all found variables for debugging them
	Returns:
		tuple: sim_lat, sim_amp -- latencies and amplitudes per slice
	"""
	# form EES stimulations indexes (in simulators begin from 0)
	stim_indexes = list(range(0, len(voltages), int(25 / step)))
	# calculate the latencies and amplitudes
	amplitudes, peaks_number, max_times, min_times, max_values, min_values = \
		__process(latencies, voltages, stim_indexes, step, ees_end, debugging, inhibition_zero=inhibition_zero,
		          after_latencies=after_latencies, first_kink=first_kink)
	# change the step
	return latencies, amplitudes, peaks_number, max_times, min_times, max_values, min_values


def find_mins(data_array): # matching_criteria was None
=======
def read_data(filepath):
>>>>>>> origin/master
	"""
	ToDo add info
	Args:
		filepath:

	Returns:

	"""
	with hdf5.File(filepath, 'r') as file:
		data_by_test = [test_values[:] for test_values in file.values()]
		if not all(map(len, data_by_test)):
			raise Exception("hdf5 has an empty data!")
	return data_by_test


def read_bio_data(path):
	"""
	Function for reading of bio data from txt file
	Args:
		path: string
			path to file

	Returns:
	 	data_RMG :list
			readed data from the first till the last stimulation,
		shifted_indexes: list
			stimulations from the zero
	"""
	with open(path) as file:
		# skipping headers of the file
		for i in range(6):
			file.readline()
		reader = csv.reader(file, delimiter='\t')
		# group elements by column (zipping)
		grouped_elements_by_column = list(zip(*reader))
		# avoid of NaN data
		raw_data_RMG = [float(x) if x != 'NaN' else 0 for x in grouped_elements_by_column[2]]
		# FixMe use 5 if new data else 7
		data_stim = [float(x) if x != 'NaN' else 0 for x in grouped_elements_by_column[7]]
	# preprocessing: finding minimal extrema an their indexes
	mins, indexes = find_mins(data_stim)
	# remove raw data before the first EES and after the last (slicing)
	data_RMG = raw_data_RMG[indexes[0]:indexes[-1]]
	# shift indexes to be normalized with data RMG (because a data was sliced) by value of the first EES
	shifted_indexes = [d - indexes[0] for d in indexes]

	return data_RMG, shifted_indexes


def extract_data(path, beg=None, end=None, step_from=None, step_to=None):
	"""
	ToDo add info
	Args:
		path:
		beg:
		end:
		step_from:
		step_to:

	Returns:
		np.ndarray: sliced data
	"""
	#
	if beg is None:
		beg = 0
	if end is None:
		end = int(10e6)
	# to convert
	if step_from == step_to or not step_from or not step_to:
		shrink_step = 1
	else:
		shrink_step = int(step_to / step_from)

	# slice data and shrink if need
	sliced_data = [data[beg:end][::shrink_step] for data in read_data(path)]
	# check if all lengths are equal
	if len(set(map(len, sliced_data))) <= 1:
		return np.array(sliced_data)
	raise Exception("Length of slices not equal!")


def draw_vector(p0, p1, color):
	"""
	Small function for drawing vector with arrow by two points
	Args:
		p0 (np.ndarray): begin of the vector
		p1 (np.ndarray): end of the vector
		color (str): the color of vector
	"""
	ax = plt.gca()
	# this plot is fixing the problem of hiding annotations because of their not markers origin
	ax.plot(p1[0], p1[1], '.', alpha=0)
	ax.annotate('', p1, p0, arrowprops=dict(facecolor=color, linewidth=1.0))


def center_data_by_line(y_points, debugging=False):
	"""
	Straight the data and center the rotated points cloud at (0, 0)
	Args:
		y_points (list or np.ndarray): list of Y points value
		debugging (bool): True -- will print debugging info and plot figures
	Returns:
		np.ndarray: new straighten Y data
	"""
	X = 0
	Y = 1
	# prepare original data (convert to 2D ndarray)
	dots_2D = np.stack((range(len(y_points)), y_points), axis=1)
	# calc PCA for dots cloud
	pca = PCA(n_components=2)
	pca.fit(dots_2D)
	# get PCA components for finding an rotation angle
	cos_theta = pca.components_[0, 0]
	sin_theta = pca.components_[0, 1]
	# one possible value of Theta that lies in [0, pi]
	arccos = np.arccos(cos_theta)

	# if arccos is in Q1 (quadrant), rotate CLOCKwise by arccos
	if cos_theta > 0 and sin_theta > 0:
		arccos *= -1
	# elif arccos is in Q2, rotate COUNTERclockwise by the complement of theta
	elif cos_theta < 0 and sin_theta > 0:
		arccos = np.pi - arccos
	# elif arccos is in Q3, rotate CLOCKwise by the complement of theta
	elif cos_theta < 0 and sin_theta < 0:
		arccos = -(np.pi - arccos)
	# if arccos is in Q4, rotate COUNTERclockwise by theta, i.e. do nothing
	elif cos_theta > 0 and sin_theta < 0:
		pass

	# manually build the counter-clockwise rotation matrix
	rotation_matrix = np.array([[np.cos(arccos), -np.sin(arccos)],
	                            [np.sin(arccos), np.cos(arccos)]])
	# apply rotation to each row of 'array_dots' (@ is a matrix multiplication)
	rotated_dots_2D = (rotation_matrix @ dots_2D.T).T
	# center the rotated point cloud at (0, 0)
	rotated_dots_2D -= rotated_dots_2D.mean(axis=0)

	# plot debugging figures
	if debugging:
		plt.figure(figsize=(16, 9))
		plt.suptitle("PCA")
		# plot all dots and connect them
		plt.scatter(dots_2D[:, X], dots_2D[:, Y], alpha=0.2)
		plt.plot(dots_2D[:, X], dots_2D[:, Y])
		# plot vectors
		for v_length, vector in zip(pca.explained_variance_, pca.components_):
			v = vector * 3 * np.sqrt(v_length)
			draw_vector(pca.mean_, pca.mean_ + v)
		# figure properties
		plt.tight_layout()
		plt.show()
		plt.close()

		plt.figure(figsize=(16, 9))
		plt.suptitle("Centered data")
		# plot ogignal data on centered
		plt.plot(range(len(dots_2D)), dots_2D[:, Y], label='original')
		plt.plot(range(len(rotated_dots_2D)), rotated_dots_2D[:, Y], label='centered')
		plt.axhline(y=0, color='g', linestyle='--')
		# figure properties
		plt.tight_layout()
		plt.legend()
		plt.show()

	return rotated_dots_2D[:, 1]


def prepare_data(dataset):
	"""
	Center the data set, then normalize it and return
	Args:
		dataset (np.ndarray): original dataset
	Returns:
		np.ndarray: prepared dataset per test
	"""
	prepared_data = []
	for data_per_test in dataset:
		centered_data = center_data_by_line(data_per_test)
		normalized_data = normalization(centered_data, save_centering=True)
		prepared_data.append(normalized_data)
	return prepared_data


def auto_prepare_data(folder, filename, step_size_to=None):
	"""
	ToDo add info
	Args:
		folder (str):
		filename (str):
		step_size_to (float):
	Returns:
		np.ndarray: prepared data for analysis
		int: EES frequency in Hz
	"""
	log.info(f"prepare {filename}")

	# map of cms <-> number of slices
	e_slices_number = {'6': 30, '15': 12, '13.5': 12, '21': 6}

	# extract common meta info from the filename
	ees_hz = int(filename[:filename.find('Hz')].split('_')[-1])
	if ees_hz not in [5] + list(range(10, 101, 10)):
		raise Exception("EES frequency not in allowed list")

	speed = filename[:filename.find('cms')].split('_')[-1]
	if speed not in ("6", "13.5", "15", "21"):
		raise Exception("Speed not in allowed list")

	step_size = float(filename[:filename.find('step')].split('_')[-1])
	if step_size not in (0.025, 0.1, 0.25):
		raise Exception("Step size not in allowed list")

	slice_in_steps = int(25 / step_size)
	filepath = f"{folder}/{filename}"

	# extract data of extensor
	if '_E_' in filename:
		# extract dataset based on slice numbers (except biological data)
		if 'bio_' in filename:
			dataset = extract_data(filepath, step_from=step_size, step_to=step_size_to)
		else:
			e_begin = 0
			e_end = e_slices_number[speed] * slice_in_steps
			# use native funcion for get needful data
			dataset = extract_data(filepath, e_begin, e_end, step_size, step_size_to)
	# extract data of flexor
	elif '_F_' in filename:
		# preapre flexor data
		if 'bio_' in filename:
			dataset = extract_data(filepath, step_from=step_size, step_to=step_size_to)
		else:
			f_begin = e_slices_number[speed] * slice_in_steps
			f_end = f_begin + (7 if '4pedal' in filename else 5) * slice_in_steps
			# use native funcion for get needful data
			dataset = extract_data(filepath, f_begin, f_end, step_size, step_size_to)
	# in another case
	else:
<<<<<<< HEAD
		for i in range(len(thresholds)):
			necessary_values.append(min_difference[i] + thresholds[i])
		necessary_indexes = []
		for slice in range(len(all_mins)):
			for dot in range(min_difference_indexes[slice], len(all_mins[slice])):
				if diffs[slice][dot] > necessary_values[slice]:
					vars.append(diffs[slice][dot])
					necessary_indexes.append(dot)
					break
	return min_difference_indexes, max_difference_indexes, necessary_indexes


def absolute_sum(data_list, step):
	all_bio_slices = []
	dots_per_slice = 0
	if step == 0.25:
		dots_per_slice = 100
	if step == 0.025:
		dots_per_slice = 1000
	# forming list for the plot
	for k in range(len(data_list)):
		bio_slices = []
		offset = 0
		for i in range(int(len(data_list[k]) / dots_per_slice)):
			bio_slices_tmp = []
			for j in range(offset, offset + dots_per_slice):
				bio_slices_tmp.append(data_list[k][j])
			bio_slices.append(normalization(bio_slices_tmp, -1, 1))
			offset += dots_per_slice
		all_bio_slices.append(bio_slices)  # list [4][16][100]
	all_bio_slices = list(zip(*all_bio_slices))  # list [16][4][100]

	instant_mean = []
	for slice in range(len(all_bio_slices)):
		instant_mean_sum = []
		for dot in range(len(all_bio_slices[slice][0])):
			instant_mean_tmp = []
			for run in range(len(all_bio_slices[slice])):
				instant_mean_tmp.append(abs(all_bio_slices[slice][run][dot]))
			instant_mean_sum.append(sum(instant_mean_tmp))
		instant_mean.append(instant_mean_sum)

	volts = []
	for i in instant_mean:
		for j in i:
			volts.append(j)

	return volts


def changing_peaks(data, herz, step, max_amp_coef=-0.3, min_amp_coef=-0.5, filtering=False):
	# print("data = ", len(data), type(data))
	ees_end= 36
	latencies, amplitudes = get_lat_amp(data, herz, step)
	print("amplitudes = ", amplitudes)
	proceed_data = []
	max_times_amp = []
	max_values_amp = []
	min_times_amp = []
	min_values_amp = []

	for i in range(len(data)):
		data[i] = smooth(data[i], 7)
		proceed_data.append(sim_process(latencies, data[i], step, ees_end, inhibition_zero=True, after_latencies=True))
		max_times_amp.append(proceed_data[i][3])
		max_values_amp.append(proceed_data[i][4])
		min_times_amp.append(proceed_data[i][5])
		min_values_amp.append(proceed_data[i][6])

	if filtering:
		print("latencies = ", latencies)
		print("amplitudes = ", amplitudes)
		print("max_times_amp = ", max_times_amp)
		print("max_values_amp = ", max_values_amp)
		print("min_times_amp = ", min_times_amp)
		print("min_values_amp = ", min_values_amp)
		max_amp_in_sl = []
		min_amp_in_sl = []
		max_indexes = []

		for sl in amplitudes:
			try:
				max_amp_in_sl.append(max(sl))
				min_amp_in_sl.append(min(sl))
				max_indexes.append(sl.index(max(sl)))
			except ValueError:
				continue

		max_amp = max(max_amp_in_sl)
		min_amp = min(min_amp_in_sl)

		threshold_max = max_amp * max_amp_coef
		threshold_min = min_amp * min_amp_coef

		print("threshold_max = ", threshold_max)
		corr_ampls_max = []
		indexes_max = []
		for index, sl in enumerate(max_values_amp):
			print("max values amp = ", sl)
			corr_ampls_max_sl = []
			indexes_sl = []
			for ind_dot, dot in enumerate(sl):
				if dot > threshold_max:
					corr_ampls_max_sl.append(dot)
					indexes_sl.append(max_times_amp[index][ind_dot])
			corr_ampls_max.append(corr_ampls_max_sl)
			indexes_max.append(indexes_sl)

		corr_ampls_min = []
		indexes_min = []

		print()
		print("threshold_min = ", threshold_min)

		for index, sl in enumerate(min_values_amp):
			print("min values amp = ", sl)
			corr_ampls_min_sl = []
			indexes_sl = []
			for ind_dot, dot in enumerate(sl):
				if dot < threshold_min:
					corr_ampls_min_sl.append(dot)
					indexes_sl.append(min_times_amp[index][ind_dot])
			corr_ampls_min.append(corr_ampls_min_sl)
			indexes_min.append(indexes_sl)

		starts_from = []
		for sl in range(len(indexes_min)):
			try:
				if indexes_max[sl][0] < indexes_min[sl][0]:
					starts_from.append('max')
				else:
					starts_from.append('min')
			except IndexError:
				if len(indexes_max[sl]) == 0:
					starts_from.append('min')
				elif len(indexes_min[sl]) == 0:
					starts_from.append('max')

		print("starts_from = ", starts_from)
		print("indexes_max = ", indexes_max)
		print("indexes_min = ", indexes_min)

		for sl in range(len(indexes_min)):
			if starts_from[sl] == 'min':
				to_delete = []

				# print("sl = ", sl)
				# print("len(indexes_max[{}]) = ".format(sl), len(indexes_max[sl]))
				# print("len(indexes_min[{}]) = ".format(sl), len(indexes_min[sl]))

				if len(indexes_max[sl]) > 0 and len(indexes_min[sl]) > 0:
					for i in range(min((len(indexes_min[sl])), len(indexes_max[sl]))):
						if indexes_max[sl][i] - indexes_min[sl][i] == 1:
							to_delete.append(i)

				print("sl = ", sl)
				print("to_delete = ", to_delete)
				for i in to_delete:
					del indexes_min[sl][i]
					del corr_ampls_min[sl][i]
					print("deleted indexes_max[{}][{}]".format(sl, i))

				if len(indexes_min[sl]) == 1 and len(indexes_max[sl]) > 1:
					del indexes_max[sl][1:]
					del corr_ampls_max[sl][1:]
				if len(indexes_max[sl]) == 1 and len(indexes_min[sl]) > 1:
					del indexes_min[sl][1:]
					del corr_ampls_min[sl][1:]

				for i in range(len(indexes_min[sl])):
					try:
						if indexes_min[sl][i] > indexes_max[sl][i]:
							del indexes_max[sl][i]
							del corr_ampls_max[sl][i]
					except IndexError:
						continue

				print("indexes_max = ", indexes_max)
				print("indexes_min = ", indexes_min)

				to_delete_sl = []
				to_delete_dot = []
				if len(indexes_min[sl]) > 1:
					for i in range(min((len(indexes_min[sl]) - 1), len(indexes_max[sl]))):
						# print("indexes_min[{}][{}] = ".format(sl, i + 1), indexes_min[sl][i + 1])
						# print("indexes_max[{}][{}] = ".format(sl, i), indexes_max[sl][i])
						# print()
						if indexes_min[sl][i + 1] < indexes_max[sl][i]:
							# print("del indexes_min[{}][{}] = ".format(sl, i + 1), indexes_min[sl][i + 1])
							to_delete_sl.append(sl)
							to_delete_dot.append(i + 1)

				print("to_delete sl= ", to_delete_sl)
				print("to_delete_dot = ", to_delete_dot)
				for i in range(len(to_delete_sl)- 1, -1, -1):
					del indexes_min[to_delete_sl[i]][to_delete_dot[i]]
					del corr_ampls_min[to_delete_sl[i]][to_delete_dot[i]]

				if len(indexes_max[sl]) > len(indexes_min[sl]) and len(indexes_min[sl]) > 0:
					del indexes_max[sl][len(indexes_min[sl]):]
					del corr_ampls_max[sl][len(indexes_min[sl]):]
				else:
					if len(indexes_max[sl]) > len(indexes_min[sl]) and len(indexes_min[sl]) == 0:
						del indexes_max[sl][1:]
						del corr_ampls_max[sl][1:]

				if len(indexes_min[sl]) > len(indexes_max[sl]) and len(indexes_max[sl]) > 0 \
						and indexes_min[sl][-1] < indexes_max[sl][-1]:
					del indexes_min[sl][-1]
					del corr_ampls_min[sl][-1]

			print()
			print("indexes_max = ", indexes_max)
			print("indexes_min = ", indexes_min)
		for sl in range(len(indexes_max)):
			if starts_from[sl] == 'max':
				if len(indexes_min[sl]) == 0:
					del indexes_max[sl][1:]
					del corr_ampls_max[sl][1:]
				to_delete = []
				for i in range(len(indexes_min[sl])):
					# print("indexes_min[{}][{}] = ".format(sl, i), indexes_min[sl][i])
					# print("indexes_max[{}][{}] = ".format(sl, i), indexes_max[sl][i])
					if indexes_min[sl][i] - indexes_max[sl][i] == 1:
						to_delete.append(i)

				for i in to_delete:
					del indexes_min[sl][i]
					del corr_ampls_min[sl][i]
				if len(indexes_min[sl]) == 1 and len(indexes_max[sl]) > 1:
					del indexes_max[sl][1:]
					del corr_ampls_max[sl][1:]

				print()
				print("indexes_max = ", indexes_max)
				print("indexes_min  ", indexes_min)

				to_delete_sl = []
				to_delete_dot = []

				if len(indexes_max[sl]) > 1:
					for i in range(min((len(indexes_min[sl])), len(indexes_max[sl]))):
						# print("indexes_max[{}][{}] = ".format(sl, i + 1), indexes_max[sl][i + 1])
						# print("indexes_min[{}][{}] = ".format(sl, i), indexes_min[sl][i])
						# print()
						if indexes_max[sl][i + 1] < indexes_min[sl][i]:
							# print(" del indexes_max[{}][{}] = ".format(sl, i + 1), indexes_max[sl][i + 1])
							to_delete_sl.append(sl)
							to_delete_dot.append(i + 1)

				for i in range(len(to_delete_sl)- 1, -1, -1):
					del indexes_max[to_delete_sl[i]][to_delete_dot[i]]
					del corr_ampls_max[to_delete_sl[i]][to_delete_dot[i]]

				print()
				print("indexes_max = ", indexes_max)
				print("indexes_min  ", indexes_min)
				print()
				if len(indexes_max[sl]) > len(indexes_min[sl]) and len(indexes_min[sl]) > 0 \
						and indexes_max[sl][-1] < indexes_min[sl][-1]:
					del indexes_max[sl][-1]
					del corr_ampls_max[sl][-1]

				print("ya")
				print("indexes_max = ", indexes_max)
				print("indexes_min  ", indexes_min)
				print()

				if len(indexes_max[sl]) > len(indexes_min[sl]) + 1 and len(indexes_min[sl]) > 0:
					del indexes_max[sl][len(indexes_min[sl]) + 1:]
					del corr_ampls_max[sl][len(indexes_min[sl]) + 1:]

				print("sl = ", sl)
				print("here")
				print("indexes_max = ", indexes_max)
				print("indexes_min  ", indexes_min)

	indexes_min = list(map(list, zip(*min_times_amp)))
	indexes_max = list(map(list, zip(*max_times_amp)))

	for j in range(len(indexes_min)):
		for d in range(len(indexes_min[j])):
			indexes_min[j][d] = [i * bio_step for i in indexes_min[j][d]]

	for j in range(len(indexes_max)):
		for d in range(len(indexes_max[j])):
			indexes_max[j][d] = [i * bio_step for i in indexes_max[j][d]]

	max_peaks = []
	for run in indexes_max:
		max_peaks_tmp = []
		for ind in run:
			max_peaks_tmp.append(len(ind))
		max_peaks.append(max_peaks_tmp)

	min_peaks = []
	for run in indexes_min:
		min_peaks_tmp = []
		for ind in run:
			min_peaks_tmp.append(len(ind))
		min_peaks.append(min_peaks_tmp)
=======
		raise Exception("Couldn't parse filename and extract muscle name")
>>>>>>> origin/master

	# centering and normalizing data
	prepared_data = prepare_data(dataset)

	return prepared_data, ees_hz, step_size
