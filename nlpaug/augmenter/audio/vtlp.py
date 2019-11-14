"""
    Augmenter that apply vocal tract length perturbation (VTLP) operation to audio.
"""

from nlpaug.augmenter.audio import AudioAugmenter
import nlpaug.model.audio as nma
from nlpaug.util import Action


class VtlpAug(AudioAugmenter):
    # https://pdfs.semanticscholar.org/3de0/616eb3cd4554fdf9fd65c9c82f2605a17413.pdf
    """
    :param tuple zone: Assign a zone for augmentation. Default value is (0.2, 0.8) which means that no any
        augmentation will be applied in first 20% and last 20% of whole audio.
    :param float coverage: Portion of augmentation. Value should be between 0 and 1. If `1` is assigned, augment
        operation will be applied to target audio segment. For example, the audio duration is 60 seconds while
        zone and coverage are (0.2, 0.8) and 0.7 respectively. 42 seconds ((0.8-0.2)*0.7*60) audio will be
        augmented.
    :param int factor: Range of applying speed adjustment operation. Default value is (0.5, 2)
        Factor for time stretch. Audio will be slowing down if value is between 0 and 1.
    :param int fhi: Boundary frequency. Default value is 4800.
    :param str name: Name of this augmenter

    >>> import nlpaug.augmenter.audio as naa
    >>> aug = naa.VtlpAug()
    """

    def __init__(self, sampling_rate, zone=(0.2, 0.8), coverage=0.1, duration=None, fhi=4800,
                 factor=(0.9, 1.1), name='Vtlp_Aug', verbose=0):
        super().__init__(
            action=Action.SUBSTITUTE, name=name, device='cpu', verbose=verbose)

        self.model = self.get_model(sampling_rate, zone, coverage, duration, factor, fhi)

    @classmethod
    def get_model(cls, sampling_rate, zone, coverage, duration, factor, fhi):
        return nma.Vtlp(sampling_rate=sampling_rate, zone=zone, coverage=coverage,
                        duration=duration, factor=factor, fhi=fhi)
