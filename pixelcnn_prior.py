import numpy as np
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax
import json
import wandb
from tqdm import tqdm
from datasets import MiniImagenet
from modules import VQVAE, GatedPixelCNN


def train(data_loader, model, prior, optimizer, args):
    progress_bar = tqdm(data_loader, desc="Training")
    for images, labels in progress_bar:
        images = jnp.array(images)
        labels = jnp.array(labels)

        latents = model.apply(optimizer.target, images)
        logits = prior.apply(optimizer.target, latents, labels)
        logits = jnp.transpose(logits, (0, 2, 3, 1))

        loss = optax.softmax_cross_entropy(logits.reshape(-1, args.k), latents.reshape(-1)).mean()
        grads = jax.grad(lambda params: loss)(optimizer.target)
        optimizer = optimizer.apply_gradient(grads)

        progress_bar.set_postfix(loss=loss.item())
        wandb.log({'loss/train': loss.item()})

    return optimizer


def test(data_loader, model, prior, args):
    loss = 0.
    progress_bar = tqdm(data_loader, desc="Validation")
    for images, labels in progress_bar:
        images = jnp.array(images)
        labels = jnp.array(labels)

        latents = model.apply(optimizer.target, images)
        logits = prior.apply(optimizer.target, latents, labels)
        logits = jnp.transpose(logits, (0, 2, 3, 1))
        loss += optax.softmax_cross_entropy(logits.reshape(-1, args.k), latents.reshape(-1)).mean()

    loss /= len(data_loader)
    wandb.log({'loss/valid': loss.item()})
    return loss.item()


def main(args):
    wandb.init(project="vqvae_pixelcnn_jax")
    wandb.config.update(vars(args))

    # Transform and dataset loading
    train_dataset = MiniImagenet(args.data_folder, train=True)
    valid_dataset = MiniImagenet(args.data_folder, valid=True)
    test_dataset = MiniImagenet(args.data_folder, test=True)

    train_loader = jax.tree_map(lambda x: x.batch(args.batch_size), train_dataset)
    valid_loader = jax.tree_map(lambda x: x.batch(args.batch_size), valid_dataset)
    test_loader = jax.tree_map(lambda x: x.batch(16), test_dataset)

    model = VQVAE(in_channels=3, hidden_channels=args.hidden_size_vae, latent_dim=128,
                 num_residual_layers=4, residual_hidden_channels=32, num_embeddings=args.k, commitment_cost=1.0)
    prior = GatedPixelCNN(input_dim=args.k, dim=args.hidden_size_prior, n_layers=args.num_layers)

    params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 3, 128, 128)))['params']
    optimizer = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optax.adam(args.lr)
    )

    best_loss = float('inf')
    for epoch in range(args.num_epochs):
        optimizer = train(train_loader, model, prior, optimizer, args)
        loss = test(valid_loader, model, prior, args)
        if loss < best_loss:
            best_loss = loss
            with open(f'./models/{args.output_folder}/best.pkl', 'wb') as f:
                f.write(model.to_bytes(optimizer.params))
        print(f"Epoch {epoch+1} - Loss: {loss}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='PixelCNN Prior for VQ-VAE with JAX')
    parser.add_argument('--data-folder', type=str, default='datasets/miniimagenet', help='data folder')
    parser.add_argument('--hidden-size-vae', type=int, default=256, help='VAE hidden size')
    parser.add_argument('--hidden-size-prior', type=int, default=64, help='PixelCNN prior hidden size')
    parser.add_argument('--k', type=int, default=512, help='number of latent vectors')
    parser.add_argument('--num-layers', type=int, default=15, help='number of layers')
    parser.add_argument('--batch-size', type=int, default=128, help='batch size')
    parser.add_argument('--num-epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')
    parser.add_argument('--output-folder', type=str, default='prior', help='output folder')
    args = parser.parse_args()
    main(args)
