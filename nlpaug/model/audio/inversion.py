import numpy as np

from nlpaug.model.audio import Audio

class PolarityInversion(Audio):
	# https://en.wikipedia.org/wiki/Phase_inversion
	def manipulate(self, data, start_pos, end_pos):
		aug_data = data.copy()
		aug_data[start_pos:end_pos] = -aug_data[start_pos:end_pos]

		return aug_data
