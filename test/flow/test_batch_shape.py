import pytest

import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.flow as naf


def make_augmenter(kind):
    if kind == 'word':
        return naw.RandomWordAug(
            action='substitute', aug_p=1, aug_max=None, stopwords=['keep'])
    return nac.RandomCharAug(
        action='substitute', candidates=['x'], min_char=1,
        aug_char_p=1, aug_char_max=None, aug_word_p=1, aug_word_max=None,
        stopwords=['keep'])


def make_flow(kind, augmenter):
    if kind == 'sequential':
        return naf.Sequential([augmenter])
    if kind == 'sometimes':
        return naf.Sometimes([augmenter], aug_p=1)
    return naf.Sequential([naf.Sometimes([augmenter], aug_p=1)])


@pytest.mark.parametrize('flow_kind', ['sequential', 'sometimes', 'nested'])
@pytest.mark.parametrize('augmenter_kind', ['word', 'char'])
@pytest.mark.parametrize('n', [1, 7])
@pytest.mark.parametrize('num_thread', [1, 2])
def test_batch_keeps_one_output_per_input(flow_kind, augmenter_kind, n, num_thread):
    data = ['alpha', 'keep', 'beta', 'alpha', '']
    original = list(data)
    expected = (['_', 'keep', '_', '_', ''] if augmenter_kind == 'word'
                else ['xxxxx', 'keep', 'xxxx', 'xxxxx', ''])
    augmenter = make_augmenter(augmenter_kind)
    assert augmenter.augment(data, n=n, num_thread=num_thread) == expected

    flow = make_flow(flow_kind, augmenter)
    assert flow.augment(data, n=n, num_thread=num_thread) == expected
    assert data == original


@pytest.mark.parametrize('flow_kind', ['sequential', 'sometimes', 'nested'])
@pytest.mark.parametrize('num_thread', [1, 2])
def test_single_input_still_uses_n(flow_kind, num_thread):
    flow = make_flow(flow_kind, make_augmenter('word'))
    assert flow.augment('alpha', n=3, num_thread=num_thread) == ['_', '_', '_']


@pytest.mark.parametrize('num_thread', [1, 2])
def test_batch_composition_preserves_rows(num_thread):
    flow = naf.Sequential([
        naf.Sometimes([make_augmenter('word')], aug_p=1),
        naf.Sequential([make_augmenter('char')]),
    ])
    assert flow.augment(['alpha', 'keep', 'beta', '', 'alpha'], num_thread=num_thread) == [
        'x', 'keep', 'x', '', 'x']


@pytest.mark.parametrize('num_thread', [1, 2])
def test_disabled_and_empty_flows_preserve_batch(num_thread):
    data = ['alpha', '', 'alpha']
    for flow in [naf.Sequential(), naf.Sometimes([make_augmenter('word')], aug_p=0)]:
        assert flow.augment(data, n=7, num_thread=num_thread) == data
        assert flow.augment([], n=7, num_thread=num_thread) == []


def test_batch_with_progress_bar():
    flow = make_flow('sequential', make_augmenter('word'))
    assert flow.augment(['alpha', 'keep', ''], progress_bar=True) == ['_', 'keep', '']
