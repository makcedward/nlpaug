import unittest
import sys
import logging


if __name__ == '__main__':
    sys.path.append('../nlpaug')

    runner = unittest.TextTestRunner()

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
        # 'test/profiling/sentence/',
    ]
    
    for test_dir in test_dirs:
       loader = unittest.TestLoader()
       suite = loader.discover(test_dir)
       runner.run(suite)

    suites = []
    # suites.append(unittest.TestLoader().loadTestsFromName('augmenter.test_base_augmenter'))
    # suites.append(unittest.TestLoader().loadTestsFromName('util.text.test_tokenizer'))
    # suites.append(unittest.TestLoader().loadTestsFromName('util.selection.test_filtering'))

    # suites.append(unittest.TestLoader().loadTestsFromName('augmenter.test_text_augmenter'))

    # suites.append(unittest.TestLoader().loadTestsFromName('augmenter.sentence.test_sentence'))
    # suites.append(unittest.TestLoader().loadTestsFromName('augmenter.sentence.test_context_word_embs_sentence'))
    # suites.append(unittest.TestLoader().loadTestsFromName('augmenter.sentence.test_abst_summ'))
    # suites.append(unittest.TestLoader().loadTestsFromName('augmenter.sentence.test_lambada'))
    # suites.append(unittest.TestLoader().loadTestsFromName('augmenter.sentence.test_random'))

    # suites.append(unittest.TestLoader().loadTestsFromName('augmenter.word.test_word'))
    # suites.append(unittest.TestLoader().loadTestsFromName('augmenter.word.test_tfidf'))
    # suites.append(unittest.TestLoader().loadTestsFromName('augmenter.word.test_spelling'))
    # suites.append(unittest.TestLoader().loadTestsFromName('augmenter.word.test_antonym'))
    # suites.append(unittest.TestLoader().loadTestsFromName('augmenter.word.test_synonym'))
    # suites.append(unittest.TestLoader().loadTestsFromName('augmenter.word.test_context_word_embs'))
    # suites.append(unittest.TestLoader().loadTestsFromName('augmenter.word.test_back_translation'))
    # suites.append(unittest.TestLoader().loadTestsFromName('augmenter.word.test_word_embs'))
    # suites.append(unittest.TestLoader().loadTestsFromName('augmenter.word.test_random_word'))
    # suites.append(unittest.TestLoader().loadTestsFromName('augmenter.word.test_reserved'))
    # suites.append(unittest.TestLoader().loadTestsFromName('augmenter.word.test_split'))
    # # suites.append(unittest.TestLoader().loadTestsFromName('model.word.test_word_embs_model'))

    # suites.append(unittest.TestLoader().loadTestsFromName('augmenter.char.test_char'))
    # suites.append(unittest.TestLoader().loadTestsFromName('augmenter.char.test_keyboard'))
    # suites.append(unittest.TestLoader().loadTestsFromName('augmenter.char.test_ocr'))
    # suites.append(unittest.TestLoader().loadTestsFromName('augmenter.char.test_random_char'))

    # suites.append(unittest.TestLoader().loadTestsFromName('augmenter.test_audio_augmenter'))

    # suites.append(unittest.TestLoader().loadTestsFromName('augmenter.audio.test_audio'))
    # suites.append(unittest.TestLoader().loadTestsFromName('augmenter.audio.test_crop'))
    # suites.append(unittest.TestLoader().loadTestsFromName('augmenter.audio.test_loudness'))
    # suites.append(unittest.TestLoader().loadTestsFromName('augmenter.audio.test_mask'))
    # suites.append(unittest.TestLoader().loadTestsFromName('augmenter.audio.test_noise'))
    # suites.append(unittest.TestLoader().loadTestsFromName('augmenter.audio.test_pitch'))
    # suites.append(unittest.TestLoader().loadTestsFromName('augmenter.audio.test_shift'))
    # suites.append(unittest.TestLoader().loadTestsFromName('augmenter.audio.test_speed'))
    # suites.append(unittest.TestLoader().loadTestsFromName('augmenter.audio.test_vtlp'))
    # suites.append(unittest.TestLoader().loadTestsFromName('augmenter.audio.test_normalization'))
    # suites.append(unittest.TestLoader().loadTestsFromName('augmenter.audio.test_inversion'))

    # suites.append(unittest.TestLoader().loadTestsFromName('augmenter.spectrogram.test_spectrogram'))
    # suites.append(unittest.TestLoader().loadTestsFromName('augmenter.spectrogram.test_frequency_masking'))
    # suites.append(unittest.TestLoader().loadTestsFromName('augmenter.spectrogram.test_loudness_spec'))
    # suites.append(unittest.TestLoader().loadTestsFromName('augmenter.spectrogram.test_time_masking'))

    # suites.append(unittest.TestLoader().loadTestsFromName('flow.test_flow'))
    # suites.append(unittest.TestLoader().loadTestsFromName('flow.test_sequential'))
    # suites.append(unittest.TestLoader().loadTestsFromName('flow.test_sometimes'))

    for suite in suites:
        runner.run(suite)
