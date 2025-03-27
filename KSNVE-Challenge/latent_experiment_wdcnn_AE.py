# main.py

from pkgs import funs
from pkgs import latent

from torch.optim import Adam
import torch

import numpy as np
import pandas as pd


def main(config, args):
    latent_size = args.latent_size
    model_name = 'wdcnn_ae'
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


    train_data, train_label, in_channels = latent.get_data_label_arrays(train_df, config.sample_size, config.overlap, args.z, args.fft)
    eval_data, eval_label, in_channels = latent.get_data_label_arrays(eval_df, config.sample_size, config.overlap, args.z, args.fft) 

    train_dataset = funs.KSNVEDataset(train_data, train_label)
    eval_dataset = funs.KSNVEDataset(eval_data, eval_label)

    train_loader = funs.get_dataloader(train_dataset, config.batch_size, shuffle = True)
    eval_loader = funs.get_dataloader(eval_dataset, 1, shuffle = False)

    encoder_block = latent.Conv1dBlock
    decoder_block = latent.Upsample1dBlock
    model = latent.WDCNN_AE(encoder_block, decoder_block, latent_size,
                            in_channels=in_channels, input_size=config.sample_size).to(device)
    

    optimizer = Adam(model.parameters(), lr = config.learning_rate)
    loss = torch.nn.MSELoss()

    trainer = funs.Trainer(model, loss, optimizer, device)
    train_loss_list = trainer.train(config.epoch, train_loader)
    trainer.save(config.model_root, model_name=model_name, latent_size=latent_size)

    model_path = f'{config.model_root}/{model_name}_{latent_size}.pt'
    trainer.model.load_state_dict(torch.load(model_path, weights_only=True))

    latent_vectors, fault_labels, eval_loss_list = trainer.eval(eval_loader, latent_size, config.epoch,
                                                save_result=True, csv_name=model_name, csv_root = config.result_root)
    
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
    csv_path = f"./loss/{model_name}_{latent_size}_loss.csv"
    loss_data.to_csv(csv_path, index=False)

    print(f"CSV 파일 저장 완료: {csv_path}")

    # 2d tnse
    latent.plot_tsne(config.tsne_root, latent_vectors, fault_labels, latent_size, 
                     config.seed, n_components=2, model_name=model_name, except_IR=False)
    latent.plot_tsne(config.tsne_root, latent_vectors, fault_labels, latent_size, 
                     config.seed, n_components=2, model_name=model_name, except_IR=True)
    # 3d tnse
    # latent.plot_tsne(config.tsne_root, latent_vectors, fault_labels, latent_size, 
    #                  config.seed, n_components=3, model_name=model_name, except_IR=False)
    # latent.plot_tsne(config.tsne_root, latent_vectors, fault_labels, latent_size, 
    #                  config.seed, n_components=3, model_name=model_name, except_IR=True)

   
if __name__ == '__main__' :
    args = latent.parse_arguments()
    config = funs.load_yaml('./config.yaml')

    main(config, args=args)