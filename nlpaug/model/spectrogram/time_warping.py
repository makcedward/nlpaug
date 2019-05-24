# import numpy as np
#
# from nlpaug.model import Spectrogram
#
#
# class TimeWarping(Spectrogram):
#     def __init__(self, time_warp):
#         super(TimeWarping, self).__init__()
#
#         self.time_warp = time_warp
#
#     # TODO
#     def mask(self, mel_spectrogram):
#         """
#             From: https://arxiv.org/pdf/1904.08779.pdf,
#             Time warping is applied via the function
#             sparse image warp of tensorflow. Given
#             a log mel spectrogram with t time steps, we view it
#             as an image where the time axis is horizontal and the
#             frequency axis is vertical. A random point along the
#             horizontal line passing through the center of the image
#             within the time steps (W, t - W) is to be warped
#             either to the left or right by a distance w chosen from a
#             uniform distribution from 0 to the time warp parameter
#             W along that line.
#         :return:
#         """
#
#         time_range = mel_spectrogram.shape[1]
#         self.w = np.random.randint(self.time_warp)
#
#         center_point = np.random.randint(self.time_warp, time_range-self.time_warp)
#         distance = np.random.randint(-self.w, self.w)
#
#         # self.w0 = np.random.randint(time_range - self.t)
#         #
#         # augmented_mel_spectrogram = mel_spectrogram.copy()
#         # augmented_mel_spectrogram[:, self.time_warp:self.time_range-self.time_warp] = 0
#         # return augmented_mel_spectrogram
#         return mel_spectrogram
