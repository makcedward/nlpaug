import nlpaug, transformers, torch, fairseq, nltk
from platform import python_version
import nlpaug.augmenter.audio as naa
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas

from pyinstrument import Profiler

profiler = Profiler()

def main():
	model_paths = [
	#     'distilbert-base-uncased',
	    'bert-base-uncased',
	#     'bert-base-cased',
	#     'xlnet-base-cased',
	    # 'roberta-base',
	#     'distilroberta-base'
	]	
	for model_path in model_paths:
	    print('-----------------:', model_path)
	    aug = naw.ContextualWordEmbsAug(model_path=model_path)
	    text = 'The quick brown fox jumps over the lazaaaaaaaaay dog'
	    augmented_text = aug.augment([text]*2)
	    # print(augmented_text)
	    

if __name__ == '__main__':
	print('python_version:{}'.format(python_version()))
	print('nlpaug:{}'.format(nlpaug.__version__))
	print('transformers:{}'.format(transformers.__version__))
	print('torch:{}'.format(torch.__version__))
	print('fairseq:{}'.format(fairseq.__version__))
	print('nltk:{}'.format(nltk.__version__))

	# yappi.set_clock_type("cpu") # Use set_clock_type("wall") for wall time
	# yappi.start()
	profiler.start()
	main()
	profiler.stop()
	print(profiler.output_text(unicode=True, color=True))
	# yappi.get_func_stats().print_all()
	# yappi.get_thread_stats().print_all()