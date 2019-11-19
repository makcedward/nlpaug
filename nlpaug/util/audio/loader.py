try:
    import librosa
except ImportError:
    # No installation required if not using this function
    pass


class AudioLoader:
    @staticmethod
    def load_audio(file_path):
        return librosa.load(file_path)

    @staticmethod
    def load_mel_spectrogram(file_path, n_mels=128, fmax=8000):
        audio, sampling_rate = AudioLoader.load_audio(file_path)
        return librosa.feature.melspectrogram(y=audio, sr=sampling_rate, n_mels=n_mels, fmax=fmax)
