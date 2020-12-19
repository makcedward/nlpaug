import numpy as np

from nlpaug.model.audio import Audio


class Normalization(Audio):
	def manipulate(self, data, method, start_pos, end_pos):
		aug_data = data.copy()
		if method == 'minmax':
			new_data = self._min_max(aug_data[start_pos:end_pos])
		elif method == 'max':
			new_data = self._max(aug_data[start_pos:end_pos])
		elif method == 'standard':
			new_data = self._standard(aug_data[start_pos:end_pos])

		aug_data[start_pos:end_pos] = new_data

		return aug_data

	def get_support_methods(self):
		return ['minmax', 'max', 'standard']

	def _standard(self, data):
		return (data - np.mean(data)) / np.std(data)

	def _max(self, data):
		return data / np.amax(np.abs(data))

	def _min_max(self, data):
		lower = np.amin(np.abs(data))
		return (data - lower) / (np.amax(np.abs(data)) - lower)
