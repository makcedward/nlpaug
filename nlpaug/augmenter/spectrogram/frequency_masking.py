import numpy as np

from nlpaug.augmenter.spectrogram import SpectrogramAugmenter
from nlpaug.util import Action, Logger
import nlpaug.model.spectrogram as nms


class FrequencyMaskingAug(SpectrogramAugmenter):
    """
    Augmenter that mask spectrogram based on frequency by random values.

    :param tuple zone: Default value is (0.2, 0.8). Assign a zone for augmentation. By default, no any augmentation
         will be applied in first 20% and last 20% of whole audio.
    :param float coverage: Default value is 1 and value should be between 0 and 1. Portion of augmentation. 
        If `1` is assigned, augment operation will be applied to target audio segment. For example, the audio 
        duration is 60 seconds while zone and coverage are (0.2, 0.8) and 0.7 respectively. 42 
        seconds ((0.8-0.2)*0.7*60) audio will be augmented.
    :param tuple factor: Default value is (40, 80) and value should not exceed number of mel frequency channels. 
        Factor value will be picked within the range of this tuple value. Mask range will be between 
        [0, v - factor) while v is the number of mel frequency channels.
    :param str name: Name of this augmenter

    >>> import nlpaug.augmenter.spectogram as nas
    >>> aug = nas.FrequencyMaskingAug()
    """
    def __init__(self, name='FrequencyMasking_Aug', zone=(0.2, 0.8), coverage=1., factor=(40, 80), verbose=0, 
        silence=False, stateless=True):
        super().__init__(action=Action.SUBSTITUTE, zone=zone, coverage=coverage, factor=factor, verbose=verbose, 
            name=name, silence=silence, stateless=stateless)

        if self.factor[0] < 0 and not self.silence:
            Logger.log().warning('Lower bound of factor is less than {}.'.format(0) + 
            ' It should be non-negative value. Will use 0 as lower bound.')

        self.model = nms.FrequencyMasking()

    def substitute(self, data):
        """
            https://arxiv.org/pdf/1904.08779.pdf, https://arxiv.org/pdf/2001.01401.pdf
            Frequency masking is applied so that f consecutive mel
            frequency channels [f0, f0 + f) are masked, where f is
            first chosen from a uniform distribution from 0 to the
            frequency mask parameter F, and f0 is chosen from
            [0, v - f). v is the number of mel frequency channels.
        """

        v = data.shape[0]
        if v < self.factor[1] and not self.silence:
            Logger.log().warning('Upper bound of factor is larger than {}.'.format(v) + 
            ' It should be smaller than number of frequency. Will use {} as upper bound'.format(v))

        upper_bound = self.factor[1] if v > self.factor[1] else v
        f = self.get_random_factor(high=upper_bound, dtype='int')
        f0 = np.random.randint(v - f)

        time_start, time_end = self.get_augment_range_by_coverage(data)

        if not self.stateless:
            self.v, self.f, self.f0, self.time_start, self.time_end = v, f, f0, time_start, time_end

        return self.model.manipulate(data, f=f, f0=f0, time_start=time_start, time_end=time_end)
