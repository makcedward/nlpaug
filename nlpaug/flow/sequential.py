"""
    Flow that apply augmentation sequentially.
"""

from nlpaug.util import Action
from nlpaug.flow import Pipeline


class Sequential(Pipeline):
    """
    Flow that apply augmenters sequentially.

    :param list flow: list of flow or augmenter
    :param str name: Name of this augmenter

    >>> import nlpaug.flow as naf
    >>> import nlpaug.augmenter.char as nac
    >>> import nlpaug.augmenter.word as naw
    >>> flow = naf.Sequential([nac.RandomCharAug(), naw.RandomWordAug()])
    """

    def __init__(self, flow=None, name='Sequential_Pipeline', verbose=0):
        Pipeline.__init__(self, name=name, action=Action.SEQUENTIAL, flow=flow, include_detail=False,
                          verbose=verbose)

    def draw(self):
        return True
