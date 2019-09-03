import unittest
import sys


if __name__ == '__main__':
    sys.path.append('../nlpaug')

    test_dirs = [
        'test/augmenter/char/',
        'test/augmenter/word/',
        'test/augmenter/sentence/',
        'test/augmenter/audio/',
        'test/augmenter/spectrogram/',
        'test/flow/'
    ]

    runner = unittest.TextTestRunner()

    for test_dir in test_dirs:
       loader = unittest.TestLoader()
       suite = loader.discover(test_dir)
       runner.run(suite)

    # suite = unittest.TestLoader().loadTestsFromName('augmenter.sentence.test_context_word_embs_sentence')
    # runner.run(suite)

    # suite = unittest.TestLoader().loadTestsFromName('augmenter.word.test_context_word_embs')
    # runner.run(suite)
