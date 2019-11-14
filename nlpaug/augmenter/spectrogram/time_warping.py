# from nlpaug.augmenter.spectrogram import SpectrogramAugmenter
# from nlpaug.util import Action
# import nlpaug.model.spectrogram as nms
#
#
# class TimeWarpingAug(SpectrogramAugmenter):
# https://arxiv.org/pdf/1904.08779.pdf
#     def __init__(self, time_mask, name='TimeWarpingAug_Aug'):
#         super(TimeWarpingAug, self).__init__(
#             action=Action.SUBSTITUTE, name=name, aug_p=1, aug_min=0.3)
#
#         self.model = self.get_model(time_mask)
#
#     def substitute(self, mel_spectrogram):
#         return self.model.mask(mel_spectrogram)
#
#     def get_model(self, time_mask):
#         return nms.TimeWarping(time_mask)
