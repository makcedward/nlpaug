import matplotlib.pyplot as plt
import librosa.display
import numpy as np


class VisualSpectrogram:
    @staticmethod
    def visual(title, spectrogram):
        plt.figure(figsize=(8, 4))
        librosa.display.specshow(
            librosa.power_to_db(spectrogram, ref=np.max), y_axis='mel', fmax=8000, x_axis='time')
        plt.colorbar(format='%+10.0f dB')
        plt.title(title)
        plt.tight_layout()
        plt.show()
