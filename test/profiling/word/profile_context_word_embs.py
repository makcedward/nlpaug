import datetime

def run_core():
	print(datetime.datetime.now(), 'before import')
	import nlpaug.augmenter.word as naw

	print(datetime.datetime.now(), 'before init')
	aug = naw.ContextualWordEmbsAug(model_path='bert-base-uncased', model_type="bert", use_custom_api=True)
	text = 'The quick brown fox jumps over the lazy dog.'
	print(datetime.datetime.now(), 'before augment')
	aug.augment([text] * 2)
	print(datetime.datetime.now(), 'done')

if __name__ == '__main__':
	run_core()
	