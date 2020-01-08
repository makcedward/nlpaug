import unittest
import sys
import logging


if __name__ == '__main__':
    sys.path.append('../nlpaug')

    # disable transformer's info logging
    for file_name in ['tokenization_utils', 'file_utils', 'modeling_utils', 'modeling_xlnet',
                      'configuration_utils']:
        logging.getLogger('transformers.' + file_name).setLevel(logging.ERROR)

    test_dirs = [
        'test/augmenter/char/',
        'test/augmenter/word/',
        'test/augmenter/sentence/',
        'test/augmenter/audio/',
        'test/augmenter/spectrogram/',
        'test/model/char/',
        'test/model/word/',
        'test/util/selection/',
        'test/flow/',
        'test/profiling/sentence/',
    ]
    runner = unittest.TextTestRunner()

    for test_dir in test_dirs:
       loader = unittest.TestLoader()
       suite = loader.discover(test_dir)
       runner.run(suite)

    # suite = unittest.TestLoader().loadTestsFromName('augmenter.sentence.test_context_word_embs_sentence')
    # suite = unittest.TestLoader().loadTestsFromName('augmenter.word.test_context_word_embs')
    # suite = unittest.TestLoader().loadTestsFromName('augmenter.word.test_word_embs')
    # suite = unittest.TestLoader().loadTestsFromName('augmenter.word.test_random_word')
    # suite = unittest.TestLoader().loadTestsFromName('augmenter.char.test_random_char')
    # suite = unittest.TestLoader().loadTestsFromName('augmenter.word.test_word')
    # suite = unittest.TestLoader().loadTestsFromName('util.selection.test_filtering')
    # suite = unittest.TestLoader().loadTestsFromName('augmenter.audio.test_noise')
    # suite = unittest.TestLoader().loadTestsFromName('augmenter.test_augmenter')
    # suite = unittest.TestLoader().loadTestsFromName('model.word.test_word_embs_model')
    # runner.run(suite)
