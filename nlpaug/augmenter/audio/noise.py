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
    def __init__(self, zone=(0.2, 0.8), coverage=1.,
                 color='white', noises=None, name='Noise_Aug', noise_factor=0.01, verbose=0):
        super().__init__(
            action=Action.SUBSTITUTE, name=name, device='cpu', verbose=verbose)

        if noise_factor != 0.01:
            print(WarningMessage.DEPRECATED.format('noise_factor', '0.0.12', ''))

        self.model = self.get_model(zone, coverage, color, noises)

    @classmethod
    def get_model(cls, zone, coverage, color, noises):
        return nma.Noise(zone=zone, coverage=coverage, color=color, noises=noises)
