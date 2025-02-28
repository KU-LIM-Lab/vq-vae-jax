
import jax
import jax.numpy as jnp

import tensorflow as tf
import tensorflow_datasets as tfds

import argparse

from train import train
from model import VQVAE


def main():
    parser = argparse.ArgumentParser(description="Train VQ-VAE on ImageNet or MNIST using JAX & Flax")
    parser.add_argument("--dataset", type=str, choices=["cifar10"],help="dataset")
    parser.add_argument("--batch_size", type=int, default=128, help="")
    # parser.add_argument("--train_steps", type=int, default=250000, help="")
    parser.add_argument("--n_epochs", type=int, default=640, help="")
    parser.add_argument("--ckpt_dir", type=str, default="/vq-vae/jax/checkpoints", help="")

    parser.add_argument("--lr", type=float, default=2e-4, help="")
    args = parser.parse_args()
    
    
    train(args)
    
    
    

if __name__ == "__main__":
    main()
