import json, os


class ReadUtil:
	"""
	Helper function for reading file.

	>>> from nlpaug.util.file.read import ReadUtil
	"""
	@staticmethod
	def read_json(file_path):
		"""
		:param str file_path: Path of json file

		>>> ReadUtil.read_json('file.json')

		"""
		if os.path.exists(file_path):
			try:
				with open(file_path) as f:
					return json.load(f)
			except:
				return None
		else:
			return None
