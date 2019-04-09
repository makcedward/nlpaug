from nlpaug import Augmenter
from nlpaug.augmenter.char import CharAugmenter
from nlpaug.util import Method


class Pipeline(Augmenter, list):
    def __init__(self, action, name='Pipeline', aug_min=1, aug_p=1, flow=None, epoch=1):
        Augmenter.__init__(self, name=name, method=Method.FLOW,
                           action=action, aug_min=aug_min)
        self.aug_p = aug_p
        if flow is None:
            list.__init__(self, [])
        elif isinstance(flow, (Augmenter, CharAugmenter)):
            list.__init__(self, [flow])
        elif isinstance(flow, list):
            for subflow in flow:
                if not isinstance(subflow, Augmenter):
                    raise ValueError('At least one of the flow does not belongs to Augmenter')
            list.__init__(self, flow)
        else:
            raise Exception(
                'Expected None, Augmenter or list of Augmenter while {} is passed'.format(
                    type(flow)))
        self.epoch = epoch

    # def tokenizer(self, text):
    #     if text is None or len(text) == 0:
    #         return []
    #     return text.split(' ')
    #
    # def reverse_tokenizer(self, tokens):
    #     return ' '.join(tokens)

