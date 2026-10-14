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
