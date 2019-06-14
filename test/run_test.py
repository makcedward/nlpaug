import unittest


if __name__ == '__main__':
    test_dirs = [
        'test/augmenter/char/',
        'test/augmenter/word/',
        'test/augmenter/audio/',
        'test/augmenter/spectrogram/',
        'test/flow/'
    ]

    runner = unittest.TextTestRunner()

    for test_dir in test_dirs:
       loader = unittest.TestLoader()
       suite = loader.discover(test_dir)
       runner.run(suite)

    # suite = unittest.TestLoader().loadTestsFromName('augmenter.word.test_wordembs')
    # runner.run(suite)
