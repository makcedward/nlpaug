"""
    Augmenter that apply pitch adjustment operation to audio.
"""

from nlpaug.augmenter.audio import AudioAugmenter
from nlpaug.util import Action
import nlpaug.model.audio as nma


class PitchAug(AudioAugmenter):
    """
    Augmenter that crop segment of audio by random values between crop_range variable.

    :param int sampling_rate: sampling rate of input audio
    :param tuple pitch_range: Range of applying pitch adjustment operation. Default value is (-10, 10)
        Number of half-steps that shifting audio
    :param str name: Name of this augmenter

    >>> import nlpaug.augmenter.audio as naa
    >>> aug = naa.PitchAug(sampling_rate=44010)
    """

    def __init__(self, sampling_rate, pitch_range=(-10, 10), name='Pitch_Aug', verbose=0):
        super(PitchAug, self).__init__(
            action=Action.SUBSTITUTE, name=name, verbose=verbose)
        self.model = self.get_model(sampling_rate, pitch_range)

    @classmethod
    def get_model(cls, sampling_rate, pitch_range):
        return nma.Pitch(sampling_rate, pitch_range)
