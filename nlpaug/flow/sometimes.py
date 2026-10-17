"""
    Flow that apply augmentation randomly.
"""

from nlpaug.util import Action
from nlpaug.flow import Pipeline


class Sometimes(Pipeline):
    """
    Flow that apply augmenters randomly.

    To apply several character actions at once (issue #274), pass one
    augmenter per action. A single augmenter handles one action only.

    :param list flow: list of flow or augmenter
    :param float aug_p: Percentage of pipeline will be executed. 
    :param str name: Name of this augmenter

    >>> import nlpaug.flow as naf
    >>> import nlpaug.augmenter.char as nac
    >>> import nlpaug.augmenter.word as naw
    >>> flow = naf.Sometimes([nac.RandomCharAug(), naw.RandomWordAug()])
    >>> # issue #274: substitute + insert + delete + swap together
    >>> from nlpaug.util import Action
    >>> char_flow = naf.Sometimes([
    ...     nac.RandomCharAug(action=Action.SUBSTITUTE),
    ...     nac.RandomCharAug(action=Action.INSERT),
    ...     nac.RandomCharAug(action=Action.DELETE),
    ...     nac.RandomCharAug(action=Action.SWAP),
    ... ])
    """

    def __init__(self, flow=None, name='Sometimes_Pipeline', aug_p=0.8, verbose=0):
        Pipeline.__init__(self, name=name, action=Action.SOMETIMES,
                          flow=flow, aug_p=aug_p, include_detail=False, verbose=verbose)

    def draw(self):
        return self.aug_p > self.prob()
