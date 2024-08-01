# processing.py
import librosa
import numpy 

class melspectogram:
    """mel-spectogram"""

    def __init__(self, sampling_rate=16000, n_fft=2048, hop_length=512, win_length=1024, n_mels=128, power=2) -> None:
        self.sampling_rate = sampling_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        self.power = power

    def __call__(self, x):
        mel_spec = librosa.feature.melspectrogram(
            y=x, 
            sr=self.sampling_rate, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length, 
            win_length=self.win_length, 
            n_mels=self.n_mels, 
            power=self.power
            )
        mel_spec = librosa.power_to_db(mel_spec, ref=numpy.max)
        return mel_spec