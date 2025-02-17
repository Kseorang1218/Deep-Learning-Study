# main.py

import funs
import latent

from torch.optim import Adam
import torch


def main(config, layer_size_list):
    latent_size = layer_size_list[-1]
    print(f'\nLatent space size: {latent_size}')

    funs.set_seed(config.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('\nCreating training dataframe...\n' + '-' * 40)
    train_df = funs.make_dataframe(config.train_dir)
    print('\nCreating evaluation dataframe...\n' + '-' * 40)
    eval_df = funs.make_dataframe(config.eval_dir)

    train_data, train_label = funs.get_data_label_arrays(train_df, config.sample_size, config.overlap)
    eval_data, eval_label = funs.get_data_label_arrays(eval_df, config.sample_size, config.overlap) 

    train_dataset = funs.KSNVEDataset(train_data, train_label)
    eval_dataset = funs.KSNVEDataset(eval_data, eval_label)

    train_loader = funs.get_dataloader(train_dataset, config.batch_size, shuffle = True)
    eval_loader = funs.get_dataloader(eval_dataset, 1, shuffle = False)

    model = funs.AutoEncoder(funs.LinearBlock, layer_size_list, in_channels = 2, input_size = config.sample_size).to(device)

    optimizer = Adam(model.parameters(), lr = config.learning_rate)
    loss = torch.nn.MSELoss()

    trainer = funs.Trainer(model, loss, optimizer, device)
    trainer.train(config.epoch, train_loader)
    trainer.save(config.model_root, latent_size=latent_size)

    model_path = f'{config.model_root}/model_{latent_size}.pt'
    trainer.model.load_state_dict(torch.load(model_path, weights_only=True))

    latent_vectors, fault_labels = trainer.eval(eval_loader) 
    # 2d tnse
    latent.plot_tsne(config.tsne_root, latent_vectors, fault_labels, latent_size, 
                     config.seed, n_components=2, except_IR=False)
    latent.plot_tsne(config.tsne_root, latent_vectors, fault_labels, latent_size, 
                     config.seed, n_components=2, except_IR=True)
    # 3d tnse
    latent.plot_tsne(config.tsne_root, latent_vectors, fault_labels, latent_size, 
                     config.seed, n_components=3, except_IR=False)
    latent.plot_tsne(config.tsne_root, latent_vectors, fault_labels, latent_size, 
                     config.seed, n_components=3, except_IR=True)

   
if __name__ == '__main__' :
    args = funs.parse_arguments()
    config = funs.load_yaml('./config.yaml')

    main(config, args.latent)