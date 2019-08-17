"""
    Flow that apply augmentation randomly.
"""

from nlpaug.util import Action
from nlpaug.flow import Pipeline


class Sometimes(Pipeline):
    """
    Flow that apply augmenters randomly.

    :param list flow: list of flow or augmenter
    :param str name: Name of this augmenter

    >>> import nlpaug.flow as naf
    >>> import nlpaug.augmenter.char as nac
    >>> import nlpaug.augmenter.word as naw
    >>> flow = naf.Sometimes([nac.RandomCharAug(), naw.RandomWordAug()])
    """

    # TODO: Using epoch to implement 1-to-many
    def __init__(self, flow=None, name='Sometimes_Pipeline', pipeline_p=0.2, aug_p=1, verbose=0):
        Pipeline.__init__(self, name=name, action=Action.SOMETIMES,
                          flow=flow, epoch=1, aug_min=-1, aug_p=aug_p, verbose=verbose)

        self.pipeline_p = pipeline_p

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
                if self.pipeline_p < self.prob():
                    continue

                augmented_data = aug.augment(augmented_data)

            results.append(augmented_data)

        return results[0]
