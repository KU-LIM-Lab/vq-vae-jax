import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax
import wandb
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from tensorboardX import SummaryWriter
from modules import VQVAE, to_scalar
from datasets import MiniImagenet

# Loss Function
def compute_loss(params, model, images):
    recon, vq_loss = model.apply({'params': params}, images)
    recon_loss = jnp.mean((recon - images) ** 2)
    total_loss = recon_loss + vq_loss
    return total_loss, (recon_loss, vq_loss)

# JIT-compiled training step
def train_step(state, model, images):
    def loss_fn(params):
        return compute_loss(params, model, images)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (recon_loss, vq_loss)), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss, recon_loss, vq_loss

train_step = jax.jit(train_step, static_argnames=['model'])

# JIT-compiled evaluation step
@jax.jit
def eval_step(params, model, images):
    recon, vq_loss = model.apply({'params': params}, images)
    recon_loss = jnp.mean((recon - images) ** 2)
    return recon_loss, vq_loss

# Grid helper
def make_grid(images, nrow=8, padding=2, normalize=False, value_range=None):
    images = np.array(images)  # JAX → NumPy

    if normalize:
        images = (images - images.min()) / (images.max() - images.min() + 1e-8)
        if value_range:
            images = images * (value_range[1] - value_range[0]) + value_range[0]

    # images: (B, H, W, C) → (B, C, H, W) for grid layout
    if images.ndim == 4 and images.shape[-1] in (1, 3):
        images = images.transpose(0, 3, 1, 2)

    nmaps, c, h, w = images.shape
    xmaps = min(nrow, nmaps)
    ymaps = int(np.ceil(nmaps / xmaps))

    grid = np.zeros((c, h * ymaps + padding * (ymaps - 1), w * xmaps + padding * (xmaps - 1)), dtype=np.uint8)

    for idx in range(nmaps):
        row = idx // xmaps
        col = idx % xmaps
        img = images[idx]
        if normalize:
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            img = (img * 255).clip(0, 255).astype(np.uint8)
        else:
            img = (img * 255).clip(0, 255).astype(np.uint8)

        grid[:, row * (h + padding):row * (h + padding) + h,
             col * (w + padding):col * (w + padding) + w] = img

    return grid

# Main training loop
def main(args):
    wandb.init(project="vqvae_jax", config=vars(args))
    writer = SummaryWriter(f'./logs/{args.output_folder}')

    transform = transforms.Compose([
        transforms.RandomResizedCrop((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    train_dataset = MiniImagenet(args.data_folder, train=True, transform=transform)
    valid_dataset = MiniImagenet(args.data_folder, valid=True, transform=transform)
    test_dataset = MiniImagenet(args.data_folder, test=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)

    fixed_images, _ = next(iter(test_loader))
    fixed_images = fixed_images.numpy()
    fixed_images = np.transpose(fixed_images, (0, 2, 3, 1))
    fixed_images = jax.device_put(jnp.array(fixed_images))
    # writer.add_image('original', make_grid(fixed_images, nrow=8), 0)
    writer.add_image('original', make_grid(fixed_images, nrow=8), 0, dataformats='CHW')

    model = VQVAE(
        in_channels=3,
        hidden_channels=args.hidden_size,
        latent_dim=args.k,
        num_residual_layers=2,
        residual_hidden_channels=32,
        num_embeddings=args.k,
        commitment_cost=args.beta
    )

    rng = jax.random.PRNGKey(0)
    dummy_input = jnp.ones((args.batch_size, 128, 128, 3))
    params = model.init(rng, dummy_input)['params']
    params = jax.device_put(params)

    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optax.adam(args.lr)
    )

    best_loss = float('inf')
    for epoch in range(args.num_epochs):
        for batch_idx, (images, _) in enumerate(tqdm(train_loader)):
            images = images.numpy()
            images = np.transpose(images, (0, 2, 3, 1))
            images = jax.device_put(jnp.array(images))
            state, loss, recon_loss, vq_loss = train_step(state, model, images)
            wandb.log({
                "train_loss": float(loss),
                "train_recon_loss": float(recon_loss),
                "train_vq_loss": float(vq_loss),
                "epoch": epoch + 1,
                "batch": batch_idx
            })

        recon_sum, vq_sum = 0., 0.
        for images, _ in valid_loader:
            images = images.numpy()
            images = np.transpose(images, (0, 2, 3, 1))
            images = jax.device_put(jnp.array(images))
            r_loss, v_loss = eval_step(state.params, model, images)
            recon_sum += r_loss
            vq_sum += v_loss
        recon_avg = recon_sum / len(valid_loader)
        vq_avg = vq_sum / len(valid_loader)
        wandb.log({
            "valid_recon_loss": float(recon_avg),
            "valid_vq_loss": float(vq_avg),
            "epoch": epoch + 1
        })

        recon_sample = model.apply({'params': state.params}, fixed_images)[0]
        recon_grid = make_grid(recon_sample, nrow=8)
        # writer.add_image('reconstruction', recon_grid, epoch + 1)
        writer.add_image('reconstruction', make_grid(recon_sample, nrow=8), epoch + 1, dataformats='CHW')

        if recon_avg < best_loss:
            best_loss = recon_avg
            with open(f'./models/{args.output_folder}/best.pkl', 'wb') as f:
                f.write(model.to_bytes(state.params))

        print(f"[Epoch {epoch+1}] recon_loss={recon_avg:.4f}, vq_loss={vq_avg:.4f}")

    print("Training completed!")


# CLI
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-folder', type=str, default='./miniimagenet')
    parser.add_argument('--output-folder', type=str, default='vqvae')
    parser.add_argument('--num-epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--hidden-size', type=int, default=256)
    parser.add_argument('--k', type=int, default=512)
    parser.add_argument('--beta', type=float, default=0.25)

    args = parser.parse_args()

    main(args)
