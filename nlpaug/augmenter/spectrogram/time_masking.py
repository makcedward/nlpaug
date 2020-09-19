import numpy as np

from nlpaug.augmenter.spectrogram import SpectrogramAugmenter
from nlpaug.util import Action
import nlpaug.model.spectrogram as nms


class TimeMaskingAug(SpectrogramAugmenter):
    """
    Augmenter that mask spectrogram based on frequency by random values.

    :param tuple zone: Default value is (0.2, 0.8). Assign a zone for augmentation. By default, no any augmentation
         will be applied in first 20% and last 20% of whole audio.
    :param float coverage: Default value is 1 and value should be between 0 and 1. Portion of augmentation. 
        If `1` is assigned, augment operation will be applied to target audio segment. For example, the audio 
        duration is 60 seconds while zone and coverage are (0.2, 0.8) and 0.7 respectively. 42 
        seconds ((0.8-0.2)*0.7*60) audio will be chosen for augmentation.
    :param str name: Name of this augmenter

    >>> import nlpaug.augmenter.spectogram as nas
    >>> aug = nas.TimeMaskingAug()
    """

    def __init__(self, name='TimeMasking_Aug', zone=(0.2, 0.8), coverage=1., verbose=0, 
        silence=False, stateless=True):
        super().__init__(action=Action.SUBSTITUTE, zone=zone, coverage=coverage, factor=(1, 1), verbose=verbose, 
            name=name, silence=silence, stateless=stateless)

        self.model = nms.TimeMasking()

    def substitute(self, data):
        """
            From: https://arxiv.org/pdf/1904.08779.pdf,
            Time masking is applied so that t consecutive time steps
            [t0, t0 + t) are masked, where t is first chosen from a
            uniform distribution from 0 to the time mask parameter
            T, and t0 is chosen from [0, tau - t).
        """

        tau = data.shape[1]
        t0, time_end = self.get_augment_range_by_coverage(data)
        t = self.get_random_factor(high=time_end, dtype='int')
        
        if not self.stateless:
            self.tau, self.t, self.t0 = tau, t, t0

        return self.model.manipulate(data, t=t, t0=t0)
