import numpy as np

class Model:
	@classmethod
	def sample(cls, x, num=None):
		if isinstance(x, list):
			return np.random.choice(x, size=num, replace=False)
		elif isinstance(x, int):
			return np.random.randint(0, x, size=num)