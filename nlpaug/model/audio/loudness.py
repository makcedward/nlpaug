import numpy as np

from nlpaug.model.audio import Audio


class Loudness(Audio):
    def __init__(self, loudness_factor=(0.5, 2)):
        """

        :param loudness_factor: Input data volume will be increased (decreased). Augmented value will be picked
            within the range of this tuple value. If volume will be reduced if value is between 0 and 1.
        """
        super().__init__()
        self.loudness_factor = loudness_factor

    def manipulate(self, data):
        loud = np.random.uniform(self.loudness_factor[0], self.loudness_factor[1])
        augmented_data = data * loud
        return augmented_data
