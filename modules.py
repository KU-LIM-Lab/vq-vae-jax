import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
from typing import Any
import optax
import numpy as np

def to_scalar(arr):
    if isinstance(arr, list):
        return [float(x) for x in arr]
    else:
        return float(arr)

def weights_init(key, shape, dtype=jnp.float32):
    limit = 1 / np.sqrt(shape[-1])
    return jax.random.uniform(key, shape, dtype=dtype, minval=-limit, maxval=limit)

class VectorQuantizer(nn.Module):
    num_embeddings: int
    embedding_dim: int
    beta: float

    @nn.compact
    def __call__(self, z_e):
        # z_e = jnp.transpose(z_e, (0, 2, 3, 1))
        latents_shape = z_e.shape
        flat_z_e = jnp.reshape(z_e, [-1, self.embedding_dim])

        embedding = self.param('embedding', weights_init, (self.num_embeddings, self.embedding_dim), jnp.float32)

        distances = (
            jnp.sum(flat_z_e ** 2, axis=1, keepdims=True) +
            jnp.sum(embedding ** 2, axis=1) -
            2 * jnp.matmul(flat_z_e, embedding.T)
        )

        encoding_indices = jnp.argmin(distances, axis=1)
        encodings = jax.nn.one_hot(encoding_indices, self.num_embeddings)

        z_q = jnp.matmul(encodings, embedding)
        z_q = jnp.reshape(z_q, latents_shape)

        embedding_loss = jnp.mean((z_q - jax.lax.stop_gradient(z_e)) ** 2)
        commitment_loss = jnp.mean((jax.lax.stop_gradient(z_q) - z_e) ** 2)
        vq_loss = embedding_loss + self.beta * commitment_loss

        z_q = z_e + jax.lax.stop_gradient(z_q - z_e)

        return vq_loss, z_q, encoding_indices # jnp.transpose(z_q, (0, 3, 1, 2))

class ResidualBlock(nn.Module):
    in_channels: int
    out_channels: int
    hidden_channels: int

    @nn.compact
    def __call__(self, x):
        x_res = nn.relu(x)
        x_res = nn.Conv(self.hidden_channels, (3, 3), padding='SAME')(x_res)
        x_res = nn.relu(x_res)
        x_res = nn.Conv(self.out_channels, (1, 1))(x_res)
        return x + x_res

class ResidualStack(nn.Module):
    in_channels: int
    out_channels: int
    hidden_channels: int
    num_residual_layers: int

    @nn.compact
    def __call__(self, x):
        for _ in range(self.num_residual_layers):
            x = ResidualBlock(self.in_channels, self.out_channels, self.hidden_channels)(x)
        return nn.relu(x)

class Encoder(nn.Module):
    in_channels: int
    hidden_channels: int
    latent_dim: int
    num_residual_layers: int
    residual_hidden_channels: int

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(self.hidden_channels, (4, 4), strides=(2, 2), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.Conv(self.hidden_channels, (4, 4), strides=(2, 2), padding='SAME')(x)
        x = nn.relu(x)
        x = ResidualStack(self.hidden_channels, self.hidden_channels, self.residual_hidden_channels, self.num_residual_layers)(x)
        x = nn.Conv(self.latent_dim, (1, 1))(x)
        return x

class Decoder(nn.Module):
    latent_dim: int
    hidden_channels: int
    out_channels: int
    num_residual_layers: int
    residual_hidden_channels: int

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(self.hidden_channels, (3, 3), padding='SAME')(x)
        x = nn.relu(x)
        x = ResidualStack(self.hidden_channels, self.hidden_channels, self.residual_hidden_channels, self.num_residual_layers)(x)
        x = nn.ConvTranspose(self.hidden_channels // 2, (4, 4), strides=(2, 2), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.ConvTranspose(self.out_channels, (4, 4), strides=(2, 2), padding='SAME')(x)
        return jnp.tanh(x)

class VQVAE(nn.Module):
    in_channels: int
    hidden_channels: int
    latent_dim: int
    num_residual_layers: int
    residual_hidden_channels: int
    num_embeddings: int
    commitment_cost: float

    @nn.compact
    def __call__(self, x):
        z_e = Encoder(
            self.in_channels, self.hidden_channels, self.latent_dim,
            self.num_residual_layers, self.residual_hidden_channels)(x)
        vq_loss, z_q, _ = VectorQuantizer(
            self.num_embeddings, self.latent_dim, self.commitment_cost)(z_e)
        pred = Decoder(
            self.latent_dim, self.hidden_channels, self.in_channels,
            self.num_residual_layers, self.residual_hidden_channels)(z_q)
        return pred, vq_loss

class GatedActivation(nn.Module):
    @nn.compact
    def __call__(self, x):
        x1, x2 = jnp.split(x, 2, axis=1)
        return jnp.tanh(x1) * jax.nn.sigmoid(x2)

class GatedMaskedConv2d(nn.Module):
    mask_type: str
    dim: int
    kernel: int
    residual: bool = True
    n_classes: int = 10

    @nn.compact
    def __call__(self, x_v, x_h, h):
        h_emb = nn.Embed(self.n_classes, 2 * self.dim)(h)

        kernel_shape_vert = (self.kernel, self.kernel)
        kernel_shape_horiz = (1, self.kernel)

        vert_stack = nn.Conv(2 * self.dim, kernel_shape_vert, padding='SAME')(x_v)
        horiz_stack = nn.Conv(2 * self.dim, kernel_shape_horiz, padding='SAME')(x_h)

        if self.mask_type == 'A':
            mask = jnp.ones((self.kernel, self.kernel), dtype=jnp.float32)
            mask = mask.at[-1, :].set(0)
            mask = mask.at[:, -1].set(0)
            vert_stack = vert_stack * mask
            horiz_stack = horiz_stack * mask

        out_v = GatedActivation()(vert_stack + h_emb[:, :, None, None])
        v2h = nn.Conv(2 * self.dim, (1, 1))(out_v)

        out_h = GatedActivation()(v2h + horiz_stack + h_emb[:, :, None, None])
        if self.residual:
            out_h = nn.Conv(self.dim, (1, 1))(out_h) + x_h
        else:
            out_h = nn.Conv(self.dim, (1, 1))(out_h)

        return out_v, out_h

class GatedPixelCNN(nn.Module):
    input_dim: int = 256
    dim: int = 64
    n_layers: int = 15
    n_classes: int = 10

    @nn.compact
    def __call__(self, x, label):
        x = nn.Embed(self.input_dim, self.dim)(x)
        x = jnp.transpose(x, (0, 3, 1, 2))
        x_v, x_h = x, x

        for i in range(self.n_layers):
            mask_type = 'A' if i == 0 else 'B'
            kernel = 7 if i == 0 else 3
            residual = False if i == 0 else True
            x_v, x_h = GatedMaskedConv2d(mask_type, self.dim, kernel, residual, self.n_classes)(x_v, x_h, label)

        x = nn.Conv(512, (1, 1))(x_h)
        x = nn.relu(x)
        x = nn.Conv(self.input_dim, (1, 1))(x)
        return x

    def generate(self, label, shape=(8, 8), batch_size=64):
        x = jnp.zeros((batch_size, *shape), dtype=jnp.int32)

        for i in range(shape[0]):
            for j in range(shape[1]):
                logits = self(x, label)
                probs = jax.nn.softmax(logits[:, :, i, j], axis=-1)
                sampled = jax.random.categorical(jax.random.PRNGKey(0), logits=probs)
                x = x.at[:, i, j].set(sampled)
        return x
