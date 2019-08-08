import numpy as np

from nlpaug.model.audio import Audio


class Mask(Audio):
    def __init__(self, sampling_rate, mask_range=(0.2, 0.8), mask_factor=2, mask_with_noise=True):
        """

        :param sampling_rate: sampling rate of input audio
        :param mask_range: Range of applying mask operation. Default value is (0.2, 0.8)
            It means that first 20% and last 20% of data will not be excluded from augment operation. Augment operation
            will be applied to clip of rest of 60% time.
        :param mask_factor: duration of masking period (in second)
        :param mask_with_noise: If it is True, targeting area will be replaced by noise. Otherwise, it will be
            replaced by 0.
        """
        super().__init__()
        self.sampling_rate = sampling_rate
        self.mask_range = mask_range
        self.mask_factor = mask_factor
        self.mask_with_noise = mask_with_noise

    def manipulate(self, data):
        valid_region = (int(len(data) * self.mask_range[0]), int(len(data) * self.mask_range[1]))

        start_timeframe = np.random.randint(valid_region[0], valid_region[1])
        end_timeframe = start_timeframe + self.sampling_rate * self.mask_factor

        # Mask region is larger than valid region
        if end_timeframe > valid_region[1]:
            end_timeframe = valid_region[1]

        masked_value = None
        if self.mask_with_noise:
            masked_value = np.random.randn(end_timeframe - start_timeframe)
        else:
            masked_value = [0 for _ in range(end_timeframe-start_timeframe)]

        augmented_data = data.copy()
        augmented_data[start_timeframe:end_timeframe] = masked_value

        return augmented_data
