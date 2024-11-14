# main.py

from funs.utils import load_yaml, set_seed
from funs.databuilder import make_dataframe, get_data_label_arrays
from funs.dataset import KSNVEDataset, get_dataloader


def initialize(seed: int):
    set_seed(seed)

def main(config):
    initialize(config.seed)

    train_df = make_dataframe(config.train_dir)
    train_data, train_label = get_data_label_arrays(train_df, config.sample_rate, config.overlap)

    train_dataset = KSNVEDataset(train_data, train_label)
    train_loader = get_dataloader(train_dataset, config.batch_size, shuffle = True)



if __name__ == '__main__' :
    config = load_yaml('./config.yaml')
    main(config)