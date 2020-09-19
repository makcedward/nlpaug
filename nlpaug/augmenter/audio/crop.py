"""
    Augmenter that apply cropping operation to audio.
"""

from nlpaug.augmenter.audio import AudioAugmenter
import nlpaug.model.audio as nma
from nlpaug.util import Action, WarningMessage


class CropAug(AudioAugmenter):
    """
    :param int sampling_rate: Sampling rate of input audio. Mandatory if duration is provided.
    :param tuple zone: Assign a zone for augmentation. Default value is (0.2, 0.8) which means that no any
        augmentation will be applied in first 20% and last 20% of whole audio.
    :param float coverage: Portion of augmentation. Value should be between 0 and 1. If `0.1` is assigned, augment
        operation will be applied to target audio segment. For example, the audio duration is 60 seconds while
        zone and coverage are (0.2, 0.8) and 0.7 respectively. 25.2 seconds ((0.8-0.2)*0.7*60) audio will be
        augmented.
    :param int duration: Duration of augmentation (in second). Default value is None. If value is provided. `coverage`
        value will be ignored.
    :param str name: Name of this augmenter

    >>> import nlpaug.augmenter.audio as naa
    >>> aug = naa.CropAug(sampling_rate=44010)
    """

    def __init__(self, sampling_rate=None, zone=(0.2, 0.8), coverage=0.1, duration=None, name='Crop_Aug', 
        verbose=0, stateless=True):
        super().__init__(
            action=Action.DELETE, zone=zone, coverage=coverage, duration=duration, name=name, 
            device='cpu', verbose=verbose, stateless=stateless)

        self.sampling_rate = sampling_rate
        self.model = nma.Crop()

    def delete(self, data):
        if self.duration is None:
            start_pos, end_pos = self.get_augment_range_by_coverage(data)
        else:
            start_pos, end_pos = self.get_augment_range_by_duration(data)

        if not self.stateless:
            self.start_pos = start_pos
            self.end_pos = end_pos
            
        return self.model.manipulate(data, start_pos=start_pos, end_pos=end_pos)
