from unittest.mock import Mock, call, patch

from nlpaug.model.word_dict.wordnet import WordNet


def test_read_downloads_missing_multilingual_wordnet_resources():
    model = object.__new__(WordNet)
    model.lang = 'spa'
    fake_wordnet = Mock()
    fake_wordnet.synsets.side_effect = [LookupError, []]

    with patch('nlpaug.model.word_dict.wordnet.wordnet', fake_wordnet), \
            patch('nlpaug.model.word_dict.wordnet.nltk.download') as download:
        result = model.read()

    assert result is fake_wordnet
    assert fake_wordnet.synsets.call_args_list == [
        call('testing', lang='spa'),
        call('testing', lang='spa'),
    ]
    assert download.call_args_list == [call('wordnet'), call('omw-1.4')]


def test_pos_tag_downloads_current_english_tagger_resource():
    expected = [('hello', 'NN')]

    with patch(
        'nlpaug.model.word_dict.wordnet.nltk.pos_tag',
        side_effect=[LookupError, expected],
    ) as pos_tag, patch('nlpaug.model.word_dict.wordnet.nltk.download') as download:
        result = WordNet.pos_tag(['hello'])

    assert result == expected
    assert pos_tag.call_count == 2
    download.assert_called_once_with('averaged_perceptron_tagger_eng')


def test_pos_tag_replaces_incomplete_english_tagger_resource():
    expected = [('hello', 'NN')]

    with patch(
        'nlpaug.model.word_dict.wordnet.nltk.pos_tag',
        side_effect=[FileNotFoundError, expected],
    ) as pos_tag, patch('nlpaug.model.word_dict.wordnet.nltk.download') as download:
        result = WordNet.pos_tag(['hello'])

    assert result == expected
    assert pos_tag.call_count == 2
    download.assert_called_once_with('averaged_perceptron_tagger_eng', force=True)
