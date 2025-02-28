import numpy as np
import jax
import jax.numpy as jnp

from flax import nnx
import flax.linen as nn

import optax
import orbax.checkpoint as ocp

import tensorflow as tf
import tensorflow_datasets as tfds

import wandb
from tqdm import tqdm

# from utils import get_features, calculate_fid

from model import VQVAE
from utils import get_dataset


def loss_fn(model: VQVAE, batch):
    x_recon, loss = model(batch['image'])
    # x_recon, loss = model(jnp.asarray(batch['image']))
    recon_loss = optax.squared_error(
        predictions=x_recon, targets=batch['image']
        # predictions=x_recon, targets=jnp.asarray(batch['image'])
    ).mean()
    aux = {"recon_loss": recon_loss, "vq_loss": loss}
    return recon_loss + loss, aux

@nnx.jit
def train_step(model: VQVAE, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, batch):
    """Train for a single step."""
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, losses), grads = grad_fn(model, batch)
    metrics.update(
        total_loss=loss,
        recon_loss=losses["recon_loss"],
        vq_loss=losses["vq_loss"],
        )  # In-place updates.
    optimizer.update(grads)  # In-place updates.

@nnx.jit
def eval_step(model: VQVAE, metrics: nnx.MultiMetric, batch):
    loss, losses = loss_fn(model, batch)
    metrics.update(
        total_loss=loss,
        recon_loss=losses["recon_loss"],
        vq_loss=losses["vq_loss"],
        )  # In-place updates.
    

def train(args):
    wandb.init(
        project="vqvae-cifar10", 
        # name="vqvae-cifar10",
        config=args
        )
    
    train_ds, test_ds = get_dataset(args)
    
    n_epochs = args.n_epochs
    lr = args.lr
    ckpt_dir = args.ckpt_dir
    
    # model = VQVAE(
    #     in_channel=3,
    #     hidden_dim=256, 
    #     K=512,
    #     beta=0.25,
    #     rngs=nnx.Rngs(jax.random.PRNGKey(0))
    # )
    model = VQVAE(
        in_channel=3, 
        hidden_dim=256, 
        K=512, 
        embedding_dim=10, 
        beta=0.25, 
        rngs=nnx.Rngs(jax.random.PRNGKey(0))
    )
    
    
    optimizer = nnx.Optimizer(model, optax.adam(lr))
    
    metrics = nnx.MultiMetric(        
        total_loss=nnx.metrics.Average('total_loss'),
        recon_loss=nnx.metrics.Average('recon_loss'),
        vq_loss=nnx.metrics.Average('vq_loss'),
    )
    valid_total_loss_min = 1e9
    for epoch in range(1, n_epochs + 1):
        epoch_train_ds = train_ds.shuffle(len(train_ds), seed=epoch)
        epoch_train_ds = epoch_train_ds.batch(args.batch_size, drop_remainder=True).prefetch(1)
        
        for step, batch in enumerate(tqdm(epoch_train_ds.as_numpy_iterator())):
            # Run the optimization for one step and make a stateful update to the following:
            # - The train state's model parameters
            # - The optimizer state
            # - The training loss batch metrics
            train_step(model, optimizer, metrics, batch)

        for metric, value in metrics.compute().items():
            print(f"Epoch {epoch} - train_{metric}: {value}")
            wandb.log({f'train_{metric}': value}, step=epoch)
        
        metrics.reset()
        
        # Compute the metrics on the test set after each training epoch.
        for test_batch in tqdm(test_ds.as_numpy_iterator()):
            eval_step(model, metrics, test_batch)
        
        for metric, value in metrics.compute().items():
            print(f"Epoch {epoch} - valid_{metric}: {value}")
            wandb.log({f'valid_{metric}': value}, step=epoch)
        
        valid_total_loss = metrics.compute()['total_loss']
        metrics.reset()

        if valid_total_loss < valid_total_loss_min:
            valid_total_loss_min = valid_total_loss
            
            _, state = nnx.split(model)
            checkpointer = ocp.StandardCheckpointer()
            path = ocp.test_utils.erase_and_create_empty(ckpt_dir)
            checkpointer.save(path / 'best_loss', state)
            
        