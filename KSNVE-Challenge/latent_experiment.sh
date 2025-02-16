#!/bin/bash

python latent_experiment.py --latent 4096 2048 
python latent_experiment.py --latent 4096 2048 1024
python latent_experiment.py --latent 4096 2048 1024 512
python latent_experiment.py --latent 4096 2048 1024 256
python latent_experiment.py --latent 4096 2048 1024 256 128