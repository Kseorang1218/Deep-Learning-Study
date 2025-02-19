# main.py

import pkgs
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
import torch

import pandas as pd

def main(config, layer_size_list):
    latent_size = 4
    print(f'\nLatent space size: {latent_size}')

    pkgs.funs.set_seed(config.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('\nCreating training dataframe...\n' + '-' * 40)
    train_df = pkgs.funs.make_dataframe(config.train_dir)
    print('\nCreating evaluation dataframe...\n' + '-' * 40)
    eval_df = pkgs.funs.make_dataframe(config.eval_dir)   

    sampled_eval = eval_df.sample(frac=0.5)
    train_df = pd.concat([train_df, sampled_eval], ignore_index=True)

    train_data, train_label = pkgs.funs.get_data_label_arrays(train_df, config.sample_size, config.overlap)
    eval_data, eval_label = pkgs.funs.get_data_label_arrays(eval_df, config.sample_size, config.overlap) 

    train_dataset = pkgs.funs.KSNVEDataset(train_data, train_label)
    eval_dataset = pkgs.funs.KSNVEDataset(eval_data, eval_label)

    train_loader = pkgs.funs.get_dataloader(train_dataset, config.batch_size, shuffle = True)
    eval_loader = pkgs.funs.get_dataloader(eval_dataset, 1, shuffle = False)

    model = pkgs.latent.MLP(pkgs.funs.LinearBlock, layer_size_list, in_channels = 2, input_size = config.sample_size).to(device)
    
    optimizer = Adam(model.parameters(), lr = config.learning_rate)
    loss = CrossEntropyLoss()

    trainer = pkgs.latent.Trainer(model, loss, optimizer, device, train_loader, eval_loader)
    train_loss, val_loss = trainer.train(config.epoch)
    trainer.save(config.model_root, latent_size='MLP')

    model_path = f'{config.model_root}/model_MLP.pt'
    trainer.model.load_state_dict(torch.load(model_path, weights_only=True))

    latent_vectors, fault_labels, _, predicted_labels = trainer.eval() 

    # 2d tnse
    pkgs.latent.plot_tsne(config.tsne_root, latent_vectors, fault_labels, 'MLP', 
                     config.seed, n_components=2, except_IR=False)
    pkgs.latent.plot_tsne(config.tsne_root, latent_vectors, fault_labels, 'MLP', 
                     config.seed, n_components=2, except_IR=True)
    # 3d tnse
    pkgs.latent.plot_tsne(config.tsne_root, latent_vectors, fault_labels, 'MLP', 
                     config.seed, n_components=3, except_IR=False)
    pkgs.latent.plot_tsne(config.tsne_root, latent_vectors, fault_labels, 'MLP', 
                     config.seed, n_components=3, except_IR=True)
    
    pkgs.funs.plot_loss_curve(config.tsne_root, train_loss, val_loss, "losses_MLP")

    pkgs.funs.plot_confusion_matrix(config.tsne_root, fault_labels, predicted_labels, ["normal", "ball", "inner", "outer"], "cm_MLP")
    
   
if __name__ == '__main__' :
    args = pkgs.funs.parse_arguments()
    config = pkgs.funs.load_yaml('./config.yaml')

    main(config, args.latent)