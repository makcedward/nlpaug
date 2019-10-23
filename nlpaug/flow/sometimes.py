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

    # TODO: deprecated pipeline_p, use aug_p
    def __init__(self, flow=None, name='Sometimes_Pipeline', pipeline_p=0.2, aug_p=1, verbose=0):
        Pipeline.__init__(self, name=name, action=Action.SOMETIMES,
                          flow=flow, aug_p=aug_p, verbose=verbose)

        self.pipeline_p = pipeline_p

    def draw(self):
        return self.pipeline_p > self.prob()

    # def augment(self, data, n=1):
    #     """
    #     :param data: Data for augmentation
    #     :param int n: Number of augmented output
    #     :return: Augmented data
    #
    #     >>> augmented_data = flow.augment(data)
    #     """
    #     results = []
    #
    #     for _ in range(n):
    #         augmented_data = data[:]
    #         for aug in self:
    #             if self.pipeline_p < self.prob():
    #                 continue
    #
    #             augmented_data = aug.augment(augmented_data)
    #
    #         results.append(augmented_data)
    #
    #     return results[0]
