# main.py

from funs.utils import load_yaml, set_seed
from funs.databuilder import make_dataframe, get_data_label_arrays
from funs.dataset import KSNVEDataset, get_dataloader
from funs.model import AutoEncoder
from funs.trainer import Trainer

from torch.optim import Adam
import torch


def main(config):
    # initialize
    set_seed(config.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_df = make_dataframe(config.train_dir)
    train_data, train_label = get_data_label_arrays(train_df, config.sample_size, config.overlap)

    train_dataset = KSNVEDataset(train_data, train_label)
    train_loader = get_dataloader(train_dataset, config.batch_size, shuffle = True)

    model = AutoEncoder(in_channels=2, input_size=config.sample_size).to(device)
    optimizer = Adam(model.parameters(), lr = config.learning_rate)
    loss = torch.nn.MSELoss()

    trainer = Trainer(model, loss, optimizer, device)
    trainer.train(config.epoch, train_loader)


if __name__ == '__main__' :
    config = load_yaml('./config.yaml')
    main(config)