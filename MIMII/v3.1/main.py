# main.py
from funs.download import *
from funs.databuilder import *
from funs.dataset import *
from funs.utils import *
from funs.model import STgramMFN
from funs.trainer import Trainer
from funs.loss import CrossEntropyLoss

import os
import torch

from torch.utils.data import DataLoader

config = load_yaml('./config.yaml')

set_seed(config.seed)

train_dirs = [os.path.join(config.data_dir, config.snr, train_dir) for train_dir in config.train_dirs]
val_dirs = [os.path.join(config.data_dir, config.snr, val_dir) for val_dir in config.val_dirs]
test_dirs = [os.path.join(config.data_dir, config.snr, test_dir) for test_dir in config.test_dirs]

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


# data_root_dir = "./data"
# snr = "0_dB"
# phases = ['train', 'test', 'val']


# for ma in ['fan', 'pump', 'slider', 'valve']:
#     print(ma)
#     for phase in phases:
#         data_dir = os.path.join(data_root_dir, snr, ma)
#         phase_dir = os.path.join(data_dir, phase)
        
#         normal_count = 0
#         abnormal_count = 0
        
#         if os.path.exists(phase_dir):
#             for root, _, files in os.walk(phase_dir):
#                 for file in files:
#                     if 'abnormal' in file:
#                         abnormal_count += 1
#                     else:
#                         normal_count += 1
        
#         print(f"{phase} directory:")
#         print(f"Normal files: {normal_count}")
#         print(f"Abnormal files: {abnormal_count}")
#         print("-" * 30, '\n')



df = download_mimii(config.data_dir, config.snr, config.sample_rate)
label_dic = make_meta2label(df)

train_df, val_df, test_df = split_df(df, config.val_model_id, config.test_model_id)


# meta2label_dic, label2meta_dic = metadata_to_label(train_dirs+test_dirs)
# # print(label2meta_dic)

# train_dataset = MIMIIDataset(train_file_list, meta2label_dic)
# train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

# num_class = len(label2meta_dic.keys())

# model = STgramMFN(num_class).to(device)

# optimizer = torch.optim.Adam(model.parameters(), lr=float(1e-4))
# loss = CrossEntropyLoss()

# trainer = Trainer(model=model, epoch=config.epoch, device=device, criterion=loss,optimizer=optimizer,
#                   start_valid_epoch=0, valid_interval=config.valid_interval, train_dirs=train_dirs, valid_dirs=val_dirs, test_dirs=test_dirs,
#                   meta2label_dic=meta2label_dic, transform=train_dataset.transform, snr=config.snr)

# trainer.train(train_loader=train_dataloader)
# model_path = os.path.join(f"./model/{config.snr}", f'{config.load_epoch}_checkpoint_{config.snr}.pth.tar')
# state_dict = torch.load(model_path, map_location=device, weights_only=True)['model']
# trainer.model.load_state_dict(state_dict)

# trainer.val(save=True)
# trainer.test()
