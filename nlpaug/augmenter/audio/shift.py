from nlpaug.augmenter.audio import AudioAugmenter
from nlpaug.util import Action
import nlpaug.model.audio as nma


class ShiftAug(AudioAugmenter):
    def __init__(self, sampling_rate, shift_max=3, shift_direction='both', name='Shift_Aug', verbose=0):
        """
        :param sampling_rate: SR of audio
        :param shift_max: Max shifting in second
        :param shift_direction: Either shifting to left, shifting to right or one of them
        :param name: Name of this augmenter
        """

        super(ShiftAug, self).__init__(
            action=Action.SUBSTITUTE, name=name, verbose=verbose)
        self.model = self.get_model(sampling_rate, shift_max, shift_direction)

    def get_model(self, sampling_rate, shift_max, shift_direction):
        return nma.Shift(sampling_rate, shift_max, shift_direction)