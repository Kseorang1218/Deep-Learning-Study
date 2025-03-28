#!/bin/bash

# python latent_experiment.py --latent_size_list 4096 2048 
# python latent_experiment.py --latent_size_list 4096 2048 1024 --objective one-class 
# python latent_experiment.py --latent_size_list 4096 2048 1024 --objective one-class  512 --objective one-class 
# python latent_experiment.py --latent_size_list 4096 2048 1024 --objective one-class  256 --objective one-class 
# python latent_experiment.py --latent_size_list 4096 2048 1024 --objective one-class  256 --objective one-class  128 --objective one-class 

# python latent_experiment_MLP.py --latent_size_list 4096 2048 1024 --objective one-class  256 --objective one-class  128 --objective one-class  64 32 16

# python latent_experiment_wdcnn.py --latent_size_list 4096 2048 1024 --objective one-class  256 --objective one-class  128 --objective one-class  64 32 16

# EX 1
python latent_experiment_wdcnn_AE.py --z --latent_size 2048 --objective one-class 
python latent_experiment_wdcnn_AE.py --z --latent_size 1024 --objective one-class 
python latent_experiment_wdcnn_AE.py --z --latent_size 512 --objective one-class 
python latent_experiment_wdcnn_AE.py --z --latent_size 256 --objective one-class 
python latent_experiment_wdcnn_AE.py --z --latent_size 128 --objective one-class 

# EX 2
# python latent_experiment_wdcnn_AE.py --fft --latent_size 2048 --objective one-class   
# python latent_experiment_wdcnn_AE.py --fft --latent_size 1024 --objective one-class 
# python latent_experiment_wdcnn_AE.py --fft --latent_size 512 --objective one-class 
# python latent_experiment_wdcnn_AE.py --fft --latent_size 256 --objective one-class 
# python latent_experiment_wdcnn_AE.py --fft --latent_size 128 --objective one-class 


# # EX 4
# python latent_experiment_wdcnn_AE.py --z --fft --latent_size 2048 --objective one-class   
# python latent_experiment_wdcnn_AE.py --z --fft --latent_size 1024 --objective one-class 
# python latent_experiment_wdcnn_AE.py --z --fft --latent_size 512 --objective one-class 
# python latent_experiment_wdcnn_AE.py --z --fft --latent_size 256 --objective one-class 
# python latent_experiment_wdcnn_AE.py --z --fft --latent_size 128 --objective one-class 

# python latent_experiment_wdcnn_AE_track2.py --latent_size 2048 --objective one-class   
# python latent_experiment_wdcnn_AE_track2.py --latent_size 1024 --objective one-class 
# python latent_experiment_wdcnn_AE_track2.py --latent_size 512 --objective one-class 
# python latent_experiment_wdcnn_AE_track2.py --latent_size 256 --objective one-class 
# python latent_experiment_wdcnn_AE_track2.py --latent_size 128 --objective one-class 
# python latent_experiment_wdcnn_AE_track2.py --latent_size 100
# python latent_experiment_wdcnn_AE_track2.py --latent_size 4