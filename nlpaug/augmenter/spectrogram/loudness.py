import numpy as np

from nlpaug.augmenter.spectrogram import SpectrogramAugmenter
from nlpaug.util import Action
import nlpaug.model.spectrogram as nms


class LoudnessAug(SpectrogramAugmenter):
    """
    Augmenter that change loudness on mel spectrogram by random values.

    :param tuple zone: Default value is (0.2, 0.8). Assign a zone for augmentation. By default, no any augmentation
         will be applied in first 20% and last 20% of whole audio.
    :param float coverage: Default value is 1 and value should be between 0 and 1. Portion of augmentation. 
        If `1` is assigned, augment operation will be applied to target audio segment. For example, the audio 
        duration is 60 seconds while zone and coverage are (0.2, 0.8) and 0.7 respectively. 42 
        seconds ((0.8-0.2)*0.7*60) audio will be augmented.
    :param tuple factor: Default value is (0.5, 2). Volume change value will be picked within the range of this 
        tuple value. Volume will be reduced if value is between 0 and 1. Otherwise, volume will be increased.
    :param str name: Name of this augmenter
    """
    def __init__(self, name='Loudness_Aug', zone=(0.2, 0.8), coverage=1., factor=(0.5, 2), verbose=0,
        silence=False, stateless=True):
        super().__init__(action=Action.SUBSTITUTE, zone=zone, coverage=coverage, factor=factor, 
            verbose=verbose, name=name, silence=silence, stateless=stateless)

        self.model = nms.Loudness()

    def substitute(self, data):
        # https://arxiv.org/pdf/2001.01401.pdf

        loudness_level = self.get_random_factor()
        time_start, time_end = self.get_augment_range_by_coverage(data)

        if not self.stateless:
            self.time_start, self.time_end, self.loudness_level = time_start, time_end, loudness_level

        return self.model.manipulate(data, loudness_level=loudness_level, time_start=time_start, time_end=time_end)
