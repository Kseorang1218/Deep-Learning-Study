from download import *

from torchvision import transforms
import torch.nn as nn


data_dir = "../data"
dataset_dir = "../dataset"
snr = "0_dB"

download_mimii(data_dir, snr)
split_dirs(data_dir,snr)
# print(df)

# split_dirs(dataset_dir, snr)

# target_dir = f"{data_dir}/0_dB/fan/id_00/normal"
# files = os.listdir(target_dir)
# print(files)