from __future__ import absolute_import
from nlpaug.augmenter.audio.audio_augmenter import AudioAugmenter
from nlpaug.augmenter.audio.noise import NoiseAug
from nlpaug.augmenter.audio.shift import ShiftAug
from nlpaug.augmenter.audio.speed import SpeedAug
from nlpaug.augmenter.audio.pitch import PitchAug
from nlpaug.augmenter.audio.loudness import LoudnessAug
from nlpaug.augmenter.audio.crop import CropAug
from nlpaug.augmenter.audio.mask import MaskAug
from nlpaug.augmenter.audio.vtlp import VtlpAug
from nlpaug.augmenter.audio.normalization import NormalizeAug
from nlpaug.augmenter.audio.inversion import PolarityInverseAug

__all__ = ['audio_augmenter', 'noise', 'shift', 'speed', 'pitch', 'loudness', 'crop', 'mask', 'vtlp', 
'normalization', 'inversion']
