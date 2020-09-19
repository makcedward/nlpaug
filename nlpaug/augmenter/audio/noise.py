"""
    Augmenter that apply noise injection operation to audio.
"""

from nlpaug.augmenter.audio import AudioAugmenter
import nlpaug.model.audio as nma
from nlpaug.util import Action, WarningMessage


class NoiseAug(AudioAugmenter):
    """
    :param tuple zone: Assign a zone for augmentation. Default value is (0.2, 0.8) which means that no any
        augmentation will be applied in first 20% and last 20% of whole audio.
    :param float coverage: Portion of augmentation. Value should be between 0 and 1. If `1` is assigned, augment
        operation will be applied to target audio segment. For example, the audio duration is 60 seconds while
        zone and coverage are (0.2, 0.8) and 0.7 respectively. 42 seconds ((0.8-0.2)*0.7*60) audio will be
        augmented.
    :param str color: Colors of noise. Supported 'white', 'pink', 'red', 'brown', 'brownian', 'blue', 'azure',
        'violet', 'purple' and 'random'. If 'random' is used, noise color will be picked randomly in each augment.
    :param list noises: Background noises for noise injection. You can provide more than one background noise and
        noise will be picked randomly. Expected format is list of numpy array. If this value is provided. `color`
        value will be ignored
    :param str name: Name of this augmenter

    >>> import nlpaug.augmenter.audio as naa
    >>> aug = naa.NoiseAug()
    """
    def __init__(self, zone=(0.2, 0.8), coverage=1., color='white', noises=None, name='Noise_Aug', 
        verbose=0, stateless=True):
        super().__init__(action=Action.SUBSTITUTE, zone=zone, coverage=coverage, name=name, 
            device='cpu', verbose=verbose, stateless=stateless)

        self.color = color
        self.noises = noises
        self.model = nma.Noise()

        self.model.validate(color)

    def substitute(self, data):
        start_pos, end_pos = self.get_augment_range_by_coverage(data)
        aug_segment_size = end_pos - start_pos

        noise, color = self.model.get_noise_and_color(aug_segment_size, self.noises, self.color)

        if not self.stateless:
            self.start_pos, self.end_pos, self.aug_factor = start_pos, end_pos, color

        return self.model.manipulate(data, start_pos=start_pos, end_pos=end_pos, noise=noise)
