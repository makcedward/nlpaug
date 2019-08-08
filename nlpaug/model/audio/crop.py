import numpy as np

from nlpaug.model.audio import Audio


class Corp(Audio):
    def __init__(self, sampling_rate, crop_range=(0.2, 0.8), crop_factor=2):
        """

        :param sampling_rate: sampling rate of input audio
        :param crop_range: Range of applying crop operation. Default value is (0.2, 0.8)
            It means that first 20% and last 20% of data will not be excluded from augment operation. Augment operation
            will be applied to clip of rest of 60% time.
        :param crop_factor: duration of cropping period (in second)
        """
        super().__init__()
        self.sampling_rate = sampling_rate
        self.crop_range = crop_range
        self.crop_factor = crop_factor

    def manipulate(self, data):
        valid_region = (int(len(data) * self.crop_range[0]), int(len(data) * self.crop_range[1]))

        start_timeframe = np.random.randint(valid_region[0], valid_region[1])
        end_timeframe = start_timeframe + self.sampling_rate * self.crop_factor

        # Crop region is larger than valid region
        if end_timeframe > valid_region[1]:
            end_timeframe = valid_region[1]

        augmented_data = data.copy()
        augmented_data = np.delete(augmented_data, np.s_[start_timeframe:end_timeframe])

        return augmented_data
