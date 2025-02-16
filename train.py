import numpy as np
import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
import tensorflow as tf
import tensorflow_datasets as tfds
import wandb
from tqdm import tqdm
import pickle

from model_imagenet import VQVAE as VQVAE_Imagenet
from model_mnist import VQVAE as VQVAE_MNIST
from dataset import get_imagenet_dataloader, get_mnist_dataloader
from config import imagenet_config, mnist_config
from utils import get_features, calculate_fid


class TrainState(nn.TrainState):
    vq_loss: jnp.ndarray


def train_step(state, batch):
    def loss_fn(params):
        recon, vq_loss = state.apply_fn({'params': params}, batch)
        mse_loss = jnp.mean((recon - batch) ** 2)
        return mse_loss + vq_loss, (mse_loss, vq_loss)
    
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (mse_loss, vq_loss)), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss, mse_loss, vq_loss


def train_imagenet():
    wandb.init(project="vqvae-imagenet", config=imagenet_config)
    train_ds, test_ds = get_imagenet_dataloader()
    
    model = VQVAE_Imagenet(3, 128, imagenet_config["latent_dim"], 2, 32, imagenet_config["num_embeddings"], imagenet_config["commitment_cost"])
    rng = jax.random.PRNGKey(0)
    params = model.init(rng, jnp.ones((1, 128, 128, 3)))['params']
    optimizer = optax.adam(imagenet_config["lr"])
    state = TrainState.create(apply_fn=model.apply, params=params, tx=optimizer, vq_loss=jnp.array(0.0))

    best_valid_loss = float("inf")
    for epoch in range(imagenet_config["epochs"]):
        total_train_loss, total_train_vq_loss = 0, 0
        for batch in tqdm(train_ds, desc=f"Training Epoch {epoch+1}"):
            state, train_loss, mse_loss, vq_loss = train_step(state, batch)
            total_train_loss += mse_loss
            total_train_vq_loss += vq_loss
        avg_train_loss = total_train_loss / len(train_ds)
        avg_train_vq_loss = total_train_vq_loss / len(train_ds)

        total_valid_loss, total_valid_vq_loss = 0, 0
        real_features, fake_features = [], []
        for batch in tqdm(test_ds, desc="Validating"):
            recon, valid_vq_loss = model.apply({'params': state.params}, batch)
            valid_loss = jnp.mean((recon - batch) ** 2) + valid_vq_loss
            total_valid_loss += valid_loss
            total_valid_vq_loss += valid_vq_loss
            real_features.append(get_features(batch))
            fake_features.append(get_features(recon))
        
        avg_valid_loss = total_valid_loss / len(test_ds)
        avg_valid_vq_loss = total_valid_vq_loss / len(test_ds)
        fid_score = calculate_fid(np.concatenate(real_features, axis=0), np.concatenate(fake_features, axis=0))
        
        wandb.log({"epoch": epoch + 1, "train_loss": avg_train_loss, "train_vq_loss": avg_train_vq_loss, "valid_loss": avg_valid_loss, "valid_vq_loss": avg_valid_vq_loss, "fid_score": fid_score})
        print(f"Epoch [{epoch+1}/{imagenet_config['epochs']}], Train Loss: {avg_train_loss:.4f}, Valid Loss: {avg_valid_loss:.4f}, FID: {fid_score:.2f}")
        
        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            with open(imagenet_config["checkpoint_path"], "wb") as f:
                pickle.dump(state.params, f)
            print(f"✅ Checkpoint Saved: {imagenet_config['checkpoint_path']} (Valid Loss: {best_valid_loss:.4f})")


def train_mnist():
    wandb.init(project="vqvae-mnist", config=mnist_config)
    train_ds, test_ds = get_mnist_dataloader()
    
    model = VQVAE_MNIST(1, 16, mnist_config["latent_dim"], mnist_config["num_embeddings"], mnist_config["commitment_cost"])
    rng = jax.random.PRNGKey(0)
    params = model.init(rng, jnp.ones((1, 32, 32, 1)))['params']
    optimizer = optax.adam(mnist_config["lr"])
    state = TrainState.create(apply_fn=model.apply, params=params, tx=optimizer, vq_loss=jnp.array(0.0))

    best_valid_loss = float("inf")
    for epoch in range(mnist_config["epochs"]):
        total_train_loss, total_train_vq_loss = 0, 0
        for batch in tqdm(train_ds, desc=f"Training Epoch {epoch+1}"):
            state, train_loss, mse_loss, vq_loss = train_step(state, batch)
            total_train_loss += mse_loss
            total_train_vq_loss += vq_loss
        avg_train_loss = total_train_loss / len(train_ds)
        avg_train_vq_loss = total_train_vq_loss / len(train_ds)

        total_valid_loss, total_valid_vq_loss = 0, 0
        real_features, fake_features = [], []
        for batch in tqdm(test_ds, desc="Validating"):
            recon, valid_vq_loss = model.apply({'params': state.params}, batch)
            valid_loss = jnp.mean((recon - batch) ** 2) + valid_vq_loss
            total_valid_loss += valid_loss
            total_valid_vq_loss += valid_vq_loss
            real_features.append(get_features(batch))
            fake_features.append(get_features(recon))
        
        avg_valid_loss = total_valid_loss / len(test_ds)
        avg_valid_vq_loss = total_valid_vq_loss / len(test_ds)
        fid_score = calculate_fid(np.concatenate(real_features, axis=0), np.concatenate(fake_features, axis=0))
        
        wandb.log({"epoch": epoch + 1, "train_loss": avg_train_loss, "train_vq_loss": avg_train_vq_loss, "valid_loss": avg_valid_loss, "valid_vq_loss": avg_valid_vq_loss, "fid_score": fid_score})
        print(f"Epoch [{epoch+1}/{mnist_config['epochs']}], Train Loss: {avg_train_loss:.4f}, Valid Loss: {avg_valid_loss:.4f}, FID: {fid_score:.4f}")
        
        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            with open(mnist_config["checkpoint_path"], "wb") as f:
                pickle.dump(state.params, f)
            print(f"✅ Checkpoint Saved: {mnist_config['checkpoint_path']} (Valid Loss: {best_valid_loss:.4f})")