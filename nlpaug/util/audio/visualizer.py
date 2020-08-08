try:
    import librosa
    import librosa.display
    import matplotlib.pyplot as plt
except ImportError:
    # No installation required if not using this function
    pass

import numpy as np


class AudioVisualizer:
    @staticmethod
    def wave(title, audio, sample_rate):
        plt.figure(figsize=(8, 4))
        librosa.display.waveplot(audio, sr=sample_rate)
        plt.title(title)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def freq_power(title, audio, sample_rate, aug_audio=None):
        audio_fft = np.fft.rfft(audio)
        audio_fft /= len(audio_fft)

        freq_bins = np.arange(0, len(audio_fft), 1.0) * (sample_rate * 1.0 / len(audio_fft))
        plt.plot(freq_bins / 1000, 10 * np.log10(audio_fft), color='#FF0000', linewidth=0.02)

        if aug_audio is not None:
            aug_audio_fft = np.fft.rfft(aug_audio)
            aug_audio_fft /= len(aug_audio_fft)

            aug_freq_bins = np.arange(0, len(aug_audio_fft), 1.0) * (sample_rate * 1.0 / len(aug_audio_fft))
            plt.plot(aug_freq_bins / 1000, 10 * np.log10(aug_audio_fft), color='#000000', linewidth=0.02)

        plt.title(title)
        plt.xlabel('Frequency (k Hz)')
        plt.ylabel('Power (dB)')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def spectrogram(title, spectrogram):
        plt.figure(figsize=(8, 4))
        librosa.display.specshow(
            librosa.power_to_db(spectrogram, ref=np.max), y_axis='mel', fmax=8000, x_axis='time')
        plt.colorbar(format='%+10.0f dB')
        plt.title(title)
        plt.tight_layout()
        plt.show()
