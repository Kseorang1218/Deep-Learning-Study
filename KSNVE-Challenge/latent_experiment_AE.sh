#!/bin/bash

python latent_experiment.py --latent_size_list 4096 2048 
python latent_experiment.py --latent_size_list 4096 2048 1024
python latent_experiment.py --latent_size_list 4096 2048 1024 512
python latent_experiment.py --latent_size_list 4096 2048 1024 256
python latent_experiment.py --latent_size_list 4096 2048 1024 256 128

# python latent_experiment_MLP.py --latent_size_list 4096 2048 1024 256 128 64 32 16

# python latent_experiment_wdcnn.py --latent_size_list 4096 2048 1024 256 128 64 32 16

# python latent_experiment_wdcnn_AE_track2.py --latent_size 2048  
# python latent_experiment_wdcnn_AE_track2.py --latent_size 1024
# python latent_experiment_wdcnn_AE_track2.py --latent_size 512
# python latent_experiment_wdcnn_AE_track2.py --latent_size 256
# python latent_experiment_wdcnn_AE_track2.py --latent_size 128
# python latent_experiment_wdcnn_AE_track2.py --latent_size 100
# python latent_experiment_wdcnn_AE_track2.py --latent_size 4