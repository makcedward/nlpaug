import pytest

from nlpaug.flow.pipeline import Pipeline
from nlpaug.util import Action, Method
from nlpaug import Augmenter


class DummyAugmenter(Augmenter):
    def __init__(self, suffix='x'):
        super().__init__(name='dummy', method=Method.WORD, action=Action.SUBSTITUTE, aug_min=1, aug_max=1)
        self.suffix = suffix

    @classmethod
    def clean(cls, data):
        return data

    @classmethod
    def is_duplicate(cls, dataset, data):
        return data in dataset

    def substitute(self, data):
        return f'{data}{self.suffix}'


class DummyPipeline(Pipeline):
    def draw(self):
        return True


def test_pipeline_constructor_variants_and_errors():
    aug = DummyAugmenter()
    assert len(DummyPipeline(action=Action.SUBSTITUTE, flow=None)) == 0
    assert len(DummyPipeline(action=Action.SUBSTITUTE, flow=aug)) == 1
    assert len(DummyPipeline(action=Action.SUBSTITUTE, flow=[aug])) == 1

    with pytest.raises(ValueError):
        DummyPipeline(action=Action.SUBSTITUTE, flow=[object()])

    with pytest.raises(Exception):
        DummyPipeline(action=Action.SUBSTITUTE, flow='bad')


def test_pipeline_duplicate_resolution_and_empty_result():
    aug = DummyAugmenter('1')
    pipeline = DummyPipeline(action=Action.SUBSTITUTE, flow=[aug])

    assert pipeline.get_is_duplicate_fx() is not None
    assert pipeline.augment('a', n=2) == ['a1', 'a1']

    empty_pipeline = DummyPipeline(action=Action.SUBSTITUTE, flow=[])
    assert empty_pipeline.augment('', n=1) == []
