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
        'test/util/selection/',
        'test/flow/'
    ]

    runner = unittest.TextTestRunner()

    for test_dir in test_dirs:
       loader = unittest.TestLoader()
       suite = loader.discover(test_dir)
       runner.run(suite)

    # suite = unittest.TestLoader().loadTestsFromName('augmenter.sentence.test_context_word_embs_sentence')
    # runner.run(suite)
    #
    # suite = unittest.TestLoader().loadTestsFromName('augmenter.word.test_context_word_embs')
    # runner.run(suite)
