import nlpaug
import numpy as np
from platform import python_version

def get_lib_ver():
	lib_ver = {
		'python': python_version(),
		'nlpaug': nlpaug.__version__,
		'numpy': np.__version__
	}

	try:
		import transformers
		lib_ver['transformers'] = transformers.__version__
	except:
		pass

	try:
		import torch
		lib_ver['torch'] = torch.__version__
	except:
		pass

	try:
		import fairseq
		lib_ver['fairseq'] = fairseq.__version__
	except:
		pass

	try:
		import nltk
		lib_ver['nltk'] = nltk.__version__
	except:
		pass

	return lib_ver


