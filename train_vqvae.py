import numpy as np
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax
from tqdm import tqdm
import wandb
from tensorboardX import SummaryWriter

from modules import VQVAE, VectorQuantizedVAE, to_scalar
from datasets import MiniImagenet


def train(data_loader, model, optimizer, args):
    for epoch in range(args.num_epochs):
        loop = tqdm(data_loader, desc=f"Epoch [{epoch+1}/{args.num_epochs}]", leave=True)
        for batch_idx, (images, _) in enumerate(loop):
            images = jnp.array(images)

            def loss_fn(params):
                x_tilde, vq_loss = model.apply(params, images)
                loss_recons = jnp.mean((x_tilde - images) ** 2)
                loss = loss_recons + vq_loss
                return loss, (loss_recons, vq_loss)

            grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
            (loss, (loss_recons, vq_loss)), grads = grad_fn(optimizer.target)
            optimizer = optimizer.apply_gradient(grads)

            loop.set_postfix(
                recon_loss=loss_recons.item(),
                vq_loss=vq_loss.item(),
                total_loss=loss.item()
            )

            wandb.log({
                "train_recon_loss": loss_recons.item(),
                "train_vq_loss": vq_loss.item(),
                "train_loss": loss.item(),
                "epoch": epoch + 1,
                "batch_idx": batch_idx
            })

    return optimizer


def test(data_loader, model, params, args):
    loss_recons, loss_vq, loss = 0., 0., 0.
    loop = tqdm(data_loader, desc="Validating", leave=False)
    for images, _ in loop:
        images = jnp.array(images)
        x_tilde, vq_loss = model.apply(params, images)
        loss_recons += jnp.mean((x_tilde - images) ** 2)
        loss_vq += vq_loss

    loss_recons /= len(data_loader)
    loss_vq /= len(data_loader)

    wandb.log({
        "valid_recon_loss": loss_recons.item(),
        "valid_vq_loss": loss_vq.item(),
    })

    return loss_recons.item(), loss_vq.item()


def generate_samples(images, model, params):
    images = jnp.array(images)
    x_tilde, _ = model.apply(params, images)
    return x_tilde


def main(args):
    wandb.init(project="vqvae_training_jax")
    wandb.config.update(vars(args))

    # Logger and save path
    save_filename = f'./models/{args.output_folder}'
    writer = SummaryWriter(f'./logs/{args.output_folder}')

    # Data transformation
    transform = lambda x: (x / 255.0 - 0.5) / 0.5

    # MiniImagenet dataset loading
    train_dataset = MiniImagenet(args.data_folder, train=True)
    valid_dataset = MiniImagenet(args.data_folder, valid=True)
    test_dataset = MiniImagenet(args.data_folder, test=True)

    # Data loaders
    train_loader = jax.tree_map(lambda x: x.batch(args.batch_size), train_dataset)
    valid_loader = jax.tree_map(lambda x: x.batch(args.batch_size), valid_dataset)
    test_loader = jax.tree_map(lambda x: x.batch(16), test_dataset)

    # Fixed images for visualization
    fixed_images, _ = next(iter(test_loader))
    fixed_images = jnp.array(fixed_images)
    fixed_grid = make_grid(fixed_images, nrow=8, value_range=(-1, 1), normalize=True)
    writer.add_image('original', fixed_grid, 0)

    # Model and optimizer
    model = VQVAE(
        in_channels=3, hidden_channels=args.hidden_size, latent_dim=args.k,
        num_residual_layers=2, residual_hidden_channels=32,
        num_embeddings=args.k, commitment_cost=args.beta
    )
    params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 3, 128, 128)))['params']
    optimizer = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optax.adam(args.lr)
    )

    best_loss = float('inf')
    for epoch in range(args.num_epochs):
        optimizer = train(train_loader, model, optimizer, args)
        loss_recons, loss_vq = test(valid_loader, model, optimizer.params, args)

        # Generate samples
        reconstruction = generate_samples(fixed_images, model, optimizer.params)
        grid = make_grid(reconstruction, nrow=8, value_range=(-1, 1), normalize=True)
        writer.add_image('reconstruction', grid, epoch + 1)

        # Save the model if the loss improves
        if loss_recons < best_loss:
            best_loss = loss_recons
            with open(f'{save_filename}/best.pkl', 'wb') as f:
                f.write(model.to_bytes(optimizer.params))

        print(f"Epoch {epoch + 1}/{args.num_epochs} - Loss: {loss_recons}, VQ Loss: {loss_vq}")

    print("Training complete!")
    

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='VQ-VAE JAX')
    parser.add_argument('--num-epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--num-embeddings', type=int, default=512, help='Number of embeddings')
    parser.add_argument('--embedding-dim', type=int, default=64, help='Embedding dimension')
    args = parser.parse_args()

    wandb.init(project="vqvae_jax")

    model = VQVAE(
        in_channels=3, hidden_channels=64, latent_dim=128,
        num_residual_layers=2, residual_hidden_channels=32,
        num_embeddings=args.num_embeddings, commitment_cost=0.25
    )

    optimizer = train_state.TrainState.create(
        apply_fn=model.apply,
        params=model.init(jax.random.PRNGKey(0), jnp.ones((1, 3, 32, 32)))['params'],
        tx=optax.adam(args.learning_rate)
    )

    train_loader = MiniImagenet('./data', train=True)

    optimizer = train(train_loader, model, optimizer, args)
