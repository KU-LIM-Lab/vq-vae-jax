import jax
import jax.numpy as jnp
import flax.linen as nn

class VectorQuantizer(nn.Module):
    num_embeddings: int
    embedding_dim: int
    beta: float

    def setup(self):
        self.embedding = self.param("embedding", nn.initializers.uniform(scale=1/self.num_embeddings), (self.num_embeddings, self.embedding_dim))

    def __call__(self, latents):
        latents_shape = latents.shape
        flat_latents = latents.reshape(-1, self.embedding_dim)

        dist = jnp.sum(flat_latents ** 2, axis=1, keepdims=True) + jnp.sum(self.embedding ** 2, axis=1) - 2 * jnp.dot(flat_latents, self.embedding.T)
        encoding_index = jnp.argmin(dist, axis=1)
        quantized_latents = self.embedding[encoding_index].reshape(latents_shape)

        embedding_loss = jnp.mean((quantized_latents - latents) ** 2)
        commitment_loss = jnp.mean((quantized_latents - jax.lax.stop_gradient(latents)) ** 2)
        vq_loss = embedding_loss + self.beta * commitment_loss

        quantized_latents = latents + jax.lax.stop_gradient(quantized_latents - latents)
        return vq_loss, quantized_latents

class ResidualBlock(nn.Module):
    in_channels: int
    out_channels: int
    hidden_channels: int

    def setup(self):
        self.conv1 = nn.Conv(self.hidden_channels, kernel_size=(3, 3), strides=(1, 1), padding="SAME")
        self.conv2 = nn.Conv(self.out_channels, kernel_size=(1, 1), strides=(1, 1))

    def __call__(self, x):
        return x + self.conv2(nn.relu(self.conv1(x)))

class ResidualStack(nn.Module):
    in_channels: int
    out_channels: int
    hidden_channels: int
    num_residual_layers: int

    def setup(self):
        self.layers = [ResidualBlock(self.in_channels, self.out_channels, self.hidden_channels) for _ in range(self.num_residual_layers)]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return nn.relu(x)

class Encoder(nn.Module):
    in_channels: int
    hidden_channels: int
    latent_dim: int
    num_residual_layers: int
    residual_hidden_channels: int

    def setup(self):
        self.conv1 = nn.Conv(self.hidden_channels, kernel_size=(4, 4), strides=(2, 2), padding="SAME")
        self.conv2 = nn.Conv(self.hidden_channels, kernel_size=(4, 4), strides=(2, 2), padding="SAME")
        self.residual_stack = ResidualStack(self.hidden_channels, self.hidden_channels, self.residual_hidden_channels, self.num_residual_layers)
        self.conv3 = nn.Conv(self.latent_dim, kernel_size=(1, 1), strides=(1, 1))

    def __call__(self, x):
        x = nn.relu(self.conv1(x))
        x = nn.relu(self.conv2(x))
        x = self.residual_stack(x)
        return self.conv3(x)

class Decoder(nn.Module):
    latent_dim: int
    hidden_channels: int
    out_channels: int
    num_residual_layers: int
    residual_hidden_channels: int

    def setup(self):
        self.conv1 = nn.Conv(self.hidden_channels, kernel_size=(3, 3), strides=(1, 1), padding="SAME")
        self.residual_stack = ResidualStack(self.hidden_channels, self.hidden_channels, self.residual_hidden_channels, self.num_residual_layers)
        self.conv2 = nn.ConvTranspose(self.hidden_channels // 2, kernel_size=(4, 4), strides=(2, 2), padding="SAME")
        self.conv3 = nn.ConvTranspose(self.out_channels, kernel_size=(4, 4), strides=(2, 2), padding="SAME")

    def __call__(self, x):
        x = nn.relu(self.conv1(x))
        x = self.residual_stack(x)
        x = nn.relu(self.conv2(x))
        return nn.tanh(self.conv3(x))

class VQVAE(nn.Module):
    in_channels: int
    hidden_channels: int
    latent_dim: int
    num_residual_layers: int
    residual_hidden_channels: int
    num_embeddings: int
    commitment_cost: float

    def setup(self):
        self.encoder = Encoder(self.in_channels, self.hidden_channels, self.latent_dim, self.num_residual_layers, self.residual_hidden_channels)
        self.vector_quantizer = VectorQuantizer(self.num_embeddings, self.latent_dim, self.commitment_cost)
        self.decoder = Decoder(self.latent_dim, self.hidden_channels, self.in_channels, self.num_residual_layers, self.residual_hidden_channels)

    def __call__(self, x):
        encoded = self.encoder(x)
        vq_loss, quantized = self.vector_quantizer(encoded)
        reconstructed = self.decoder(quantized)
        return reconstructed, vq_loss