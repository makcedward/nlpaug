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

    # TODO: Using epoch to implement 1-to-many
    def __init__(self, flow=None, name='Sequential_Pipeline', verbose=0):
        Pipeline.__init__(self, name=name, action=Action.SEQUENTIAL,
                          flow=flow, epoch=1, aug_min=-1, aug_p=1, verbose=verbose)

    def augment(self, data):
        """
        :param data: Data for augmentation
        :return: Augmented data

        >>> augmented_data = flow.augment(data)
        """
        results = []
        for _ in range(self.epoch):
            augmented_data = data[:]
            for aug in self:
                augmented_data = aug.augment(augmented_data)

            results.append(augmented_data)

        return results[0]
