# tutorial.py
from download import download_mimii, split_dirs
from dataset import MIMIIDataset
from utils import get_filename_list, metadata_to_label
from torch.utils.data import DataLoader
from model import STgramMFN
from trainer import Trainer
from loss import CrossEntropyLoss

import os
import torch

"""constants"""
data_dir = "../data"
train_dir = ""
snr = "6_dB"
train_dirs = [
    os.path.join(data_dir, snr, "fan/train"),
    os.path.join(data_dir, snr, "pump/train"),
    os.path.join(data_dir, snr, "slider/train"),
    os.path.join(data_dir, snr, "valve/train")
]
val_dirs = [
    os.path.join(data_dir, snr, "fan/val"),
    os.path.join(data_dir, snr, "pump/val"),
    os.path.join(data_dir, snr, "slider/val"),
    os.path.join(data_dir, snr, "valve/val")
]
test_dirs = [
    os.path.join(data_dir, snr, "fan/test"),
    os.path.join(data_dir, snr, "pump/test"),
    os.path.join(data_dir, snr, "slider/test"),
    os.path.join(data_dir, snr, "valve/test")
]
batch_size = 64 # 원 논문에서는 128이었으나..
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
epoch = 100
valid_interval = 10
load_epoch = 'best'

"""code"""
download_mimii(data_dir, snr)
split_dirs(data_dir,snr)

train_file_list = []
for train_dir in train_dirs:
    train_file_list.extend(get_filename_list(train_dir))

meta2label_dic, label2meta_dic = metadata_to_label(train_dirs+test_dirs)

train_dataset = MIMIIDataset(train_file_list, meta2label_dic)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

num_class = len(label2meta_dic.keys())

model = STgramMFN(num_class).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=float(1e-4))
loss = CrossEntropyLoss()

# files, labels = create_test_file_list("../data/0_dB/fan/val", "id_00")
# print(files)
# # print(labels)
# print(meta2label_dic)

trainer = Trainer(model=model, epoch=epoch, device=device, criterion=loss,optimizer=optimizer,
                  start_valid_epoch=0, valid_interval=valid_interval, train_dirs=train_dirs, valid_dirs=val_dirs, test_dirs=test_dirs,
                  meta2label_dic=meta2label_dic, transform=train_dataset.transform, snr=snr)

trainer.train(train_loader=train_dataloader)
model_path = os.path.join("../model", f'{load_epoch}_checkpoint_{snr}.pth.tar')
state_dict = torch.load(model_path, map_location=device, weights_only=True)['model']
trainer.model.load_state_dict(state_dict)

trainer.val()
trainer.test()
