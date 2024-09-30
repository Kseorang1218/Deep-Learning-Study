import os
import torchaudio
from torch.utils.data import Dataset, DataLoader

import utils


def wav_to_log_mel(wav_path, sr, n_fft, win_length, hop_length, n_mels, power):
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        n_mels=n_mels,
        power=power,
    )

    wav_data, _ = torchaudio.load(wav_path)
    amp_to_db = torchaudio.transforms.AmplitudeToDB()

    # noise_wav = utils.add_noise_to_wav(wav_data, -4)

    # mel_spec = mel_transform(noise_wav)
    mel_spec = mel_transform(wav_data)
    log_mel_spec = amp_to_db(mel_spec)

    return log_mel_spec


def get_train_loader(args):
    train_dir = args.train_dir
    sr = args.sr
    n_fft = args.n_fft
    win_length = args.win_length
    hop_length = args.hop_length
    n_mels = args.n_mels
    power = args.power

    file_list = os.listdir(train_dir)
    file_list.sort()
    file_list = [os.path.join(train_dir, file) for file in file_list]
    train_dataloader = BaselineDataLoader(
        file_list, sr, n_fft, win_length, hop_length, n_mels, power
    )

    train_loader = DataLoader(
        train_dataloader,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.n_workers,
    )
    return train_loader


def get_eval_loader(args):
    eval_dir = args.eval_dir
    sr = args.sr
    n_fft = args.n_fft
    win_length = args.win_length
    hop_length = args.hop_length
    n_mels = args.n_mels
    power = args.power

    file_list = os.listdir(eval_dir)
    file_list.sort()
    file_list = [os.path.join(eval_dir, file) for file in file_list]
    eval_dataloader = BaselineDataLoader(
        file_list, sr, n_fft, win_length, hop_length, n_mels, power
    )

    eval_loader = DataLoader(
        eval_dataloader, batch_size=1, shuffle=False, num_workers=0
    )
    return eval_loader, file_list


def get_test_loader(args):
    test_dir = args.test_dir
    sr = args.sr
    n_fft = args.n_fft
    win_length = args.win_length
    hop_length = args.hop_length
    n_mels = args.n_mels
    power = args.power

    file_list = os.listdir(test_dir)
    file_list.sort()
    file_list = [os.path.join(test_dir, file) for file in file_list]
    eval_dataloader = BaselineDataLoader(
        file_list, sr, n_fft, win_length, hop_length, n_mels, power
    )

    eval_loader = DataLoader(
        eval_dataloader, batch_size=1, shuffle=False, num_workers=0
    )
    return eval_loader, file_list


class BaselineDataLoader(Dataset):
    def __init__(self, data_path, sr, n_fft, win_length, hop_length, n_mels, power):
        self.data_path = data_path
        self.sr = sr
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.power = power

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, idx):
        wav_path = self.data_path[idx]
        log_mel_spec = wav_to_log_mel(
            wav_path,
            self.sr,
            self.n_fft,
            self.win_length,
            self.hop_length,
            self.n_mels,
            self.power,
        )

        anomaly_label = utils.get_fault_label(wav_path)
        return log_mel_spec, anomaly_label
