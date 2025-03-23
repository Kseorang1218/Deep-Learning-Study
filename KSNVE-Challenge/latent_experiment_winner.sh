#!/bin/bash


# dsvdd + z + one-class
# python latent_experiment_winner.py --z --objective one-class --latent_size 2048
# python latent_experiment_winner.py --z --objective one-class --latent_size 1024
python latent_experiment_winner.py --z --objective one-class --latent_size 512
# python latent_experiment_winner.py --z --objective one-class --latent_size 256
# python latent_experiment_winner.py --z --objective one-class --latent_size 128

# dsvdd + z + soft-boundary
# python latent_experiment_winner.py --z --objective soft-boundary --latent_size 2048
# python latent_experiment_winner.py --z --objective soft-boundary --latent_size 1024
# python latent_experiment_winner.py --z --objective soft-boundary --latent_size 512
# python latent_experiment_winner.py --z --objective soft-bounday --latent_size 256
# python latent_experiment_winner.py --z --objective soft-boundary --latent_size 128

# dsvdd + one-class
# python latent_experiment_winner.py --objective one-class --latent_size 2048
# python latent_experiment_winner.py --objective one-class --latent_size 1024
python latent_experiment_winner.py --objective one-class --latent_size 512
# python latent_experiment_winner.py --objective one-class --latent_size 256
# python latent_experiment_winner.py --objective one-class --latent_size 128

# dsvdd + soft-boundary
# python latent_experiment_winner.py --objective soft-boundary --latent_size 2048
# python latent_experiment_winner.py --objective soft-boundary --latent_size 1024
# python latent_experiment_winner.py --objective soft-boundary --latent_size 512
# python latent_experiment_winner.py --objective soft-boundary --latent_size 256
# python latent_experiment_winner.py --objective soft-boundary --latent_size 128


# python latent_experiment_winner.py --objective one-class
# python latent_experiment_winner.py --FFT
# python latent_experiment_winner.py --DSVDD
# python latent_experiment_winner.py --z --FFT
# python latent_experiment_winner.py --z --DSVDD
# python latent_experiment_winner.py --FFT --DSVDD
# python latent_experiment_winner.py --z --FFT --DSVDD
