import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple

class VectorQuantizer(nn.Module):
    num_embeddings: int
    embedding_dim: int
    beta: float = 0.25  # 커밋먼트 손실 가중치

    def setup(self):
        self.embedding = self.param(
            "embedding",  # 파라미터 이름
            jax.nn.initializers.uniform(scale=1.0 / self.num_embeddings),
            (self.num_embeddings, self.embedding_dim),
        )

    def __call__(self, latents: jnp.ndarray) -> Tuple[jnp.ndarray, float]:
        # [B, D, H, W] → [B, H, W, D]
        latents = jnp.transpose(latents, (0, 2, 3, 1))
        flat_latents = latents.reshape(-1, self.embedding_dim)  # [BHW, D]

        # 코드북과 L2 거리 계산
        distances = (
            jnp.sum(flat_latents ** 2, axis=1, keepdims=True)
            + jnp.sum(self.embedding ** 2, axis=1)
            - 2 * jnp.dot(flat_latents, self.embedding.T)
        )

        encoding_indices = jnp.argmin(distances, axis=1)
        encodings = jax.nn.one_hot(encoding_indices, self.num_embeddings)

        # 코드북 벡터 할당
        quantized_latents = jnp.dot(encodings, self.embedding)
        quantized_latents = quantized_latents.reshape(latents.shape)

        # VQ Loss 계산
        commitment_loss = jnp.mean((jax.lax.stop_gradient(quantized_latents) - latents) ** 2)
        embedding_loss = jnp.mean((quantized_latents - jax.lax.stop_gradient(latents)) ** 2)
        vq_loss = embedding_loss + self.beta * commitment_loss

        # Straight-Through Estimator (STE) 적용
        quantized_latents = latents + jax.lax.stop_gradient(quantized_latents - latents)

        return jnp.transpose(quantized_latents, (0, 3, 1, 2)), vq_loss
    
    
class VQVAE(nn.Module):
    in_channels: int
    embedding_dim: int
    num_embeddings: int
    hidden_dims: Tuple[int] = (128, 256)
    beta: float = 0.25

    def setup(self):
        self.encoder = nn.Sequential([
            nn.Conv(self.hidden_dims[0], kernel_size=(4, 4), strides=(2, 2)),
            nn.leaky_relu,
            nn.Conv(self.hidden_dims[1], kernel_size=(4, 4), strides=(2, 2)),
            nn.leaky_relu,
            nn.Conv(self.embedding_dim, kernel_size=(1, 1)),
        ])
        self.vq_layer = VectorQuantizer(self.num_embeddings, self.embedding_dim, self.beta)
        self.decoder = nn.Sequential([
            nn.ConvTranspose(self.hidden_dims[1], kernel_size=(4, 4), strides=(2, 2)),
            nn.leaky_relu,
            nn.ConvTranspose(self.in_channels, kernel_size=(4, 4), strides=(2, 2)),
            nn.tanh
        ])

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def sample(self, num_samples: int, rng):
        # 코드북 샘플링
        indices = jax.random.randint(rng, (num_samples,), 0, self.num_embeddings)
        quantized = self.vq_layer.embedding[indices]
        samples = self.decode(quantized)
        return samples

    def __call__(self, x):
        encoding = self.encode(x)
        quantized, vq_loss = self.vq_layer(encoding)
        reconstruction = self.decode(quantized)
        return reconstruction, vq_loss
