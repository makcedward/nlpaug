import unittest


if __name__ == '__main__':
    test_dirs = [
        'test/augmenter/char/',
        # 'test/augmenter/word/',
        # 'test/flow/'
    ]

    runner = unittest.TextTestRunner()

    for test_dir in test_dirs:
        loader = unittest.TestLoader()
        suites = loader.discover(test_dir)
        for s in suites:
            clazz = str(s)
            if 'TestWordNet' in clazz:
                runner.run(s)
