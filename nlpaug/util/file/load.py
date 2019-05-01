import librosa


class LoadUtil:
    @staticmethod
    def load_mel_spectrogram(file_path, n_mels=128, fmax=8000):
        audio, sampling_rate = librosa.load(file_path)
        return librosa.feature.melspectrogram(
            y=audio, sr=sampling_rate, n_mels=n_mels, fmax=fmax)