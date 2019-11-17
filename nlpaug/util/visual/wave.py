import matplotlib.pyplot as plt
import librosa.display
import numpy as np


class VisualWave:
    @staticmethod
    def visual(title, audio, sample_rate):
        plt.figure(figsize=(8, 4))
        librosa.display.waveplot(audio, sr=sample_rate)
        plt.title(title)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def freq_power(title, audio, sample_rate):
        audio_fft = np.fft.rfft(audio)
        audio_fft /= len(audio_fft)

        freq_bins = np.arange(0, len(audio_fft), 1.0) * (sample_rate * 1.0 / len(audio_fft))
        plt.plot(freq_bins / 1000, 10 * np.log10(audio_fft), color='#ff7f00', linewidth=0.02)
        plt.title(title)
        plt.xlabel('Frequency (k Hz)')
        plt.ylabel('Power (dB)')
        plt.tight_layout()
        plt.show()
