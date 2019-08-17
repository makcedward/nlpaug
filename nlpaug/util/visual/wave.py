import matplotlib.pyplot as plt
import librosa.display


class VisualWave:
    @staticmethod
    def visual(title, audio, sample_rate):
        plt.figure(figsize=(8, 4))
        librosa.display.waveplot(audio, sr=sample_rate)
        plt.title(title)
        plt.tight_layout()
        plt.show()
