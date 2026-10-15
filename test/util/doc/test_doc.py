from nlpaug.util import Action, Doc


def test_doc_tracks_tokens_and_change_logs():
    doc = Doc('Hello world', ['Hello', 'world'])

    assert doc.get_original_tokens() == ['Hello', 'world']
    assert doc.get_augmented_tokens() == ['Hello', 'world']
    assert doc.size() == 2
    assert doc.changed_count() == 0

    doc.add_change_log(0, new_token='Hi', action=Action.SUBSTITUTE, change_seq=2)
    doc.add_token(1, token='new', action=Action.INSERT, change_seq=1)
    doc.update_change_log(1, token='there', action=Action.INSERT, change_seq=3)

    assert doc.get_token(0).get_latest_token().token == 'Hi'
    assert doc.get_token(1).get_latest_token().token == 'there'
    assert doc.get_augmented_tokens() == ['Hi', 'there', 'world']
    assert doc.changed_count() == 1

    change_logs = doc.get_change_logs()
    assert len(change_logs) == 2
    assert change_logs[0]['new_token'] == 'Hi'
    assert change_logs[1]['new_token'] == 'there'


def test_doc_empty_tokens_starts_empty():
    doc = Doc()
    assert doc.size() == 0
    assert doc.get_original_tokens() == []
    assert doc.get_augmented_tokens() == []


def test_doc_tracks_wordpiece_and_punctuation_positions():
    text = 'testttt 123 http://curious.org'
    tokens = ['test', '##tt', '##t', '123', 'http', ':', '/', '/', 'curious', '.', 'org']
    doc = Doc(text, tokens)

    assert [doc.get_token(i).orig_token.start_pos for i in range(doc.size())] == [
        0, 4, 6, 8, 12, 16, 17, 18, 19, 26, 27
    ]

    doc.add_change_log(8, new_token='site', action=Action.SUBSTITUTE, change_seq=1)
    doc.add_change_log(10, new_token='com', action=Action.SUBSTITUTE, change_seq=2)
    change_logs = doc.get_change_logs()

    assert change_logs[0]['orig_start_pos'] == 19
    assert change_logs[0]['new_start_pos'] == 19
    assert change_logs[1]['orig_start_pos'] == 27
    assert change_logs[1]['new_start_pos'] == 24
