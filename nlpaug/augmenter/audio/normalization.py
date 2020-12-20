"""
    Augmenter that apply mask normalization to audio.
"""

import random

from nlpaug.augmenter.audio import AudioAugmenter
import nlpaug.model.audio as nma
from nlpaug.util import Action, WarningMessage


class NormalizeAug(AudioAugmenter):
    """
    :param str method: It supports 'minmax', 'max' and 'standard'. For 'minmax', data will be 
        substracted by min value in data and dividing by range of max value and min value. For
        'max', data will be divided by max value only. For 'standard', data will be substracted
        by mean value and dividing by value of standard deviation. If 'random' is used, method 
        will be picked randomly in each augment.
    :param tuple zone: Assign a zone for augmentation. Default value is (0.2, 0.8) which means that no any
        augmentation will be applied in first 20% and last 20% of whole audio.
    :param float coverage: Portion of augmentation. Value should be between 0 and 1. If `0.1` is assigned, augment
        operation will be applied to target audio segment. For example, the audio duration is 60 seconds while
        zone and coverage are (0.2, 0.8) and 0.7 respectively. 25.2 seconds ((0.8-0.2)*0.7*60) audio will be
        augmented.
    :param str name: Name of this augmenter

    >>> import nlpaug.augmenter.audio as naa
    >>> aug = naa.NormalizeAug()
    """

    def __init__(self, method='max', zone=(0.2, 0.8), coverage=0.3, name='Normalize_Aug', verbose=0, 
        stateless=True):
        super().__init__(
            action=Action.SUBSTITUTE, zone=zone, coverage=coverage, name=name, device='cpu', 
            verbose=verbose, stateless=stateless)

        self.model = nma.Normalization()
        self.method = method
        self.validate()

    def random_method(self):
        return self.sample(self.model.get_support_methods(), 1)[0]

    def substitute(self, data):
        start_pos, end_pos = self.get_augment_range_by_coverage(data)

        method = self.random_method() if self.method == 'random' else self.method
        
        if not self.stateless:
            self.start_pos = start_pos
            self.end_pos = end_pos
            self.run_method = method

        return self.model.manipulate(data, method=method, start_pos=start_pos, end_pos=end_pos)

    def validate(self):
        if self.method not in ['random'] + self.model.get_support_methods():
            raise ValueError('{} does not support yet. You may pick one of {}'.format(
                self.method, ['random'] + self.model.get_support_methods()))

        return True
