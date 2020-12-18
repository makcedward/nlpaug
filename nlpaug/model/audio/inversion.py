import numpy as np

from nlpaug.model.audio import Audio

class PolarityInversion(Audio):
	# https://en.wikipedia.org/wiki/Phase_inversion
	def manipulate(self, data):
		return data * -1
