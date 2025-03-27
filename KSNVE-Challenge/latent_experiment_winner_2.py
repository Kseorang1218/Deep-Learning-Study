# main.py

from pkgs import funs
from pkgs import latent
from pkgs import dsvdd

import torch

import os

import numpy as np
import pandas as pd


def main(config, args, pretrain=True, load_model=False):
    mode = args.objective
    latent_size = args.latent_size

    model_name = f'IG_dsvdd_{mode}_{latent_size}'

    print(f'\nLatent space size: {latent_size}')
    
    funs.set_seed(config.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('\nCreating training dataframe...\n' + '-' * 40)
    train_df = funs.make_dataframe(config.train_dir)
    print('\nCreating evaluation dataframe...\n' + '-' * 40)
    eval_df = funs.make_dataframe(config.eval_dir)


    n = len(train_df['xdata']) 
    train_df['x_fft'] = train_df['xdata'].apply(lambda x: np.abs(np.fft.fft(x)/n))
    train_df['y_fft'] = train_df['ydata'].apply(lambda y: np.abs(np.fft.fft(y)/n))
    train_df['zdata'] = train_df.apply(lambda row: np.array(row['xdata']) - np.array(row['ydata']), axis=1)
    train_df['z_fft'] = train_df['zdata'].apply(lambda z: np.abs(np.fft.fft(z)/n))

    n = len(eval_df['xdata']) 
    eval_df['x_fft'] = eval_df['xdata'].apply(lambda x: np.abs(np.fft.fft(x)/n))
    eval_df['y_fft'] = eval_df['ydata'].apply(lambda y: np.abs(np.fft.fft(y)/n))
    eval_df['zdata'] = eval_df.apply(lambda row: np.array(row['xdata']) - np.array(row['ydata']), axis=1)
    eval_df['z_fft'] = eval_df['zdata'].apply(lambda z: np.abs(np.fft.fft(z)/n))


    if args.z:
        model_name = f'{model_name}_z'
    if args.fft:
        model_name = f'{model_name}_fft'

    print(model_name)

    train_data, train_label, in_channels = latent.get_data_label_arrays(train_df, config.sample_size, config.overlap, args.z, args.fft)
    eval_data, eval_label, in_channels = latent.get_data_label_arrays(eval_df, config.sample_size, config.overlap, args.z, args.fft) 

    train_dataset = funs.KSNVEDataset(train_data, train_label)
    eval_dataset = funs.KSNVEDataset(eval_data, eval_label)

    train_loader = funs.get_dataloader(train_dataset, config.batch_size, shuffle = True)
    eval_loader = funs.get_dataloader(eval_dataset, 1, shuffle = False)


    net = dsvdd.Net(config.sample_size, latent_space_size=latent_size, in_channels=in_channels).to(device)
    # net = dsvdd.Vanilla(config.sample_size, latent_space_size=latent_size, in_channels=in_channels).to(device)
    trainer = dsvdd.DeepSVDDTrainer(objective=mode, R=0, c=None, nu=0.1, config=config, device=device)
    # ae_net = dsvdd.AE_Net(input_size=config.sample_size, latent_space_size=latent_size, in_channels=in_channels)
    # ae_trainer = dsvdd.AETrainer(config, device)


    deep_SVDD = dsvdd.DeepSVDD(objective=mode, nu=0.1)
    deep_SVDD.set_network(net)
    model_root = os.path.join(config.model_root, model_name)

    # if load_model:
    #     deep_SVDD.load_model(ae_net, model_path=model_root, load_ae=True)
    
    # if pretrain:
    #     deep_SVDD.pretrain(ae_net, ae_trainer, train_loader, eval_loader, latent_size, save_result=False)

    train_loss_list = deep_SVDD.train(trainer=trainer, train_loader=train_loader)
    latent_vectors, fault_labels, eval_loss_list = deep_SVDD.eval(trainer, eval_loader, latent_size,
                                                  csv_name=f'{model_name}', csv_root=config.result_root)
    

    max_len = max(len(train_loss_list), len(eval_loss_list))

    # 짧은 리스트에 NaN 추가하여 길이를 맞춤
    train_loss_list += [np.nan] * (max_len - len(train_loss_list))
    eval_loss_list += [np.nan] * (max_len - len(eval_loss_list))

    # DataFrame 생성
    loss_data = pd.DataFrame({
        'train_loss': train_loss_list,
        'eval_loss': eval_loss_list
    })

    # CSV 파일로 저장
    csv_path = f"./loss/{model_name}_loss.csv"
    loss_data.to_csv(csv_path, index=False)

    print(f"CSV 파일 저장 완료: {csv_path}")

    
    deep_SVDD.save_model(model_root, save_ae=False)

    # 2d tnse
    latent.plot_tsne(config.tsne_root, latent_vectors, fault_labels, latent_size, 
                     config.seed, n_components=2, model_name=model_name, except_IR=False)
    latent.plot_tsne(config.tsne_root, latent_vectors, fault_labels, latent_size, 
                     config.seed, n_components=2, model_name=model_name, except_IR=True)
    # # 3d tnse
    # latent.plot_tsne(config.tsne_root, latent_vectors, fault_labels, latent_size, 
    #                  config.seed, n_components=3, model_name=model_name, except_IR=False)
    # latent.plot_tsne(config.tsne_root, latent_vectors, fault_labels, latent_size, 
    #                  config.seed, n_components=3, model_name=model_name, except_IR=True)

   
if __name__ == '__main__' :
    args = latent.parse_arguments()
    config = funs.load_yaml('./config.yaml')

    main(config, args=args)