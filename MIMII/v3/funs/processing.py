# processing.py
import torchaudio

class melspectogram(object):
    def __init__(self, sampling_rate=16000, n_fft=1024, hop_length=512, win_length=1024, n_mels=128, power=2.0):
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sampling_rate,
            win_length=win_length,
            hop_length=hop_length,
            n_fft=n_fft,
            n_mels=n_mels,
            power=power)
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(stype='power')

    def __call__(self, x):
        return self.amplitude_to_db(self.mel_spec(x))