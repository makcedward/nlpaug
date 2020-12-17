"""
    Augmenter that apply mask normalization to audio.
"""

from nlpaug.augmenter.audio import AudioAugmenter
import nlpaug.model.audio as nma
from nlpaug.util import Action, WarningMessage


class NormalizeAug(AudioAugmenter):
    """
    :param str method: It supports 'minmax', 'max' and 'standard'. For 'minmax', data will be 
        substracted by min value in data and dividing by range of max value and min value. For
        'max', data will be divided by max value only. For 'standard', data will be substracted
        by mean value and dividing by value of standard deviation.
    :param str name: Name of this augmenter

    >>> import nlpaug.augmenter.audio as naa
    >>> aug = naa.NormalizeAug()
    """

    def __init__(self, method='max', name='Normalize_Aug', verbose=0, stateless=True):
        super().__init__(
            action=Action.SUBSTITUTE, name=name, device='cpu', verbose=verbose, 
            stateless=stateless)

        self.method = method
        self.model = nma.Normalization()
        self.validate()

    def substitute(self, data):
        return self.model.manipulate(data, method=self.method)

    def validate(self):
        if self.method not in self.model.get_support_methods():
            raise ValueError('{} does not support yet. You may pick one of {}'.format(
                self.method, self.model.get_support_methods()))

        return True
