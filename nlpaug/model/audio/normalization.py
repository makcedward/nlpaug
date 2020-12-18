import numpy as np

from nlpaug.model.audio import Audio


class Normalization(Audio):
	def manipulate(self, data, method):
		if method == 'minmax':
			return self._min_max(data)
		if method == 'max':
			return self._max(data)
		if method == 'standard':
			return self._standard(data)

		return data

	def get_support_methods(self):
		return ['minmax', 'max', 'standard']

	def _standard(self, data):
		return (data - np.mean(data)) / np.std(data)

	def _max(self, data):
		return data / np.amax(np.abs(data))

	def _min_max(self, data):
		lower = np.amin(np.abs(data))
		return (data - lower) / (np.amax(np.abs(data)) - lower)
