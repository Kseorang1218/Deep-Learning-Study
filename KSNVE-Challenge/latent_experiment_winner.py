# main.py

from pkgs import funs
from pkgs import latent
from pkgs import dsvdd

from torch.optim import Adam
import torch

import os


def main(config, latent_size, mode, pretrain=True, load_model=False):
    model_name = f'dsvdd_{mode}_{latent_size}'

    print(f'\nLatent space size: {latent_size}')
    
    funs.set_seed(config.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('\nCreating training dataframe...\n' + '-' * 40)
    train_df = funs.make_dataframe(config.train_dir)
    print('\nCreating evaluation dataframe...\n' + '-' * 40)
    eval_df = funs.make_dataframe(config.eval_dir)

    if args.z:      # x-y 데이터 전처리
        train_data, train_label = latent.get_data_label_arrays(train_df, config.sample_size, config.overlap)
        eval_data, eval_label = latent.get_data_label_arrays(eval_df, config.sample_size, config.overlap) 
        in_channels = 1
        model_name = f'{model_name}_z'
    else:            # x-y 데이터 전처리
        train_data, train_label = funs.get_data_label_arrays(train_df, config.sample_size, config.overlap)
        eval_data, eval_label = funs.get_data_label_arrays(eval_df, config.sample_size, config.overlap) 
        in_channels = 2
    print(model_name)

    train_dataset = funs.KSNVEDataset(train_data, train_label)
    eval_dataset = funs.KSNVEDataset(eval_data, eval_label)

    train_loader = funs.get_dataloader(train_dataset, config.batch_size, shuffle = True)
    eval_loader = funs.get_dataloader(eval_dataset, 1, shuffle = False)


    net = dsvdd.Net(config.sample_size, latent_space_size=latent_size, in_channels=in_channels).to(device)
    trainer = dsvdd.DeepSVDDTrainer(objective=args.objective, R=0, c=None, nu=0.1, config=config, device=device)
    ae_net = dsvdd.AE_Net(input_size=config.sample_size, latent_space_size=latent_size, in_channels=in_channels)
    ae_trainer = dsvdd.AETrainer(config, device)


    deep_SVDD = dsvdd.DeepSVDD(objective='one-class', nu=0.1)
    deep_SVDD.set_network(net)
    if load_model:
        deep_SVDD.load_model(ae_net, model_path=load_model, load_ae=True)
    
    if pretrain:
        deep_SVDD.pretrain(ae_net, ae_trainer, train_loader, eval_loader, latent_size, save_result=False)

    deep_SVDD.train(trainer=trainer, train_loader=train_loader)
    latent_vectors, fault_labels = deep_SVDD.eval(trainer, eval_loader, latent_size,
                                                  csv_name=f'{model_name}', csv_root=config.result_root)
    
    model_name = os.path.join(config.model_root, model_name)
    deep_SVDD.save_model(model_name, save_ae=False)

    # 2d tnse
    # latent.plot_tsne(config.tsne_root, latent_vectors, fault_labels, latent_size, 
    #                  config.seed, n_components=2, model_name=model_name, except_IR=False)
    # latent.plot_tsne(config.tsne_root, latent_vectors, fault_labels, latent_size, 
    #                  config.seed, n_components=2, model_name=model_name, except_IR=True)
    # # 3d tnse
    # latent.plot_tsne(config.tsne_root, latent_vectors, fault_labels, latent_size, 
    #                  config.seed, n_components=3, model_name=model_name, except_IR=False)
    # latent.plot_tsne(config.tsne_root, latent_vectors, fault_labels, latent_size, 
    #                  config.seed, n_components=3, model_name=model_name, except_IR=True)

   
if __name__ == '__main__' :
    args = latent.parse_arguments()
    config = funs.load_yaml('./config.yaml')

    main(config, args.latent_size, args.objective)