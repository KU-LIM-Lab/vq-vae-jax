import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple

class VectorQuantizer(nn.Module):
    code_book_size: int
    embedding_dim: int
    commitment_cost: float

    def setup(self):
        self.embedding = self.param("embedding", nn.initializers.uniform(scale=1/self.code_book_size), (self.code_book_size, self.embedding_dim))

    def __call__(self, inputs):
        input_shape = inputs.shape
        flat_input = inputs.reshape(-1, self.embedding_dim)
        
        distances = jnp.sum((flat_input[:, None, :] - self.embedding[None, :, :]) ** 2, axis=-1)
        encoding_indices = jnp.argmin(distances, axis=1)
        quantized = self.embedding[encoding_indices].reshape(input_shape)
        
        e_latent_loss = jnp.mean((quantized - inputs) ** 2)
        q_latent_loss = jnp.mean((quantized - jax.lax.stop_gradient(inputs)) ** 2)
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        quantized = inputs + jax.lax.stop_gradient(quantized - inputs) if self.is_mutable_collection('params') else quantized
        return loss, quantized, encoding_indices.reshape(input_shape[0], -1)

class ResBlock(nn.Module):
    channels: int

    def setup(self):
        self.norm1 = nn.GroupNorm(num_groups=8, num_channels=self.channels)
        self.conv1 = nn.Conv(self.channels, kernel_size=(3, 3), strides=(1, 1), padding=((1, 1), (1, 1)))
        self.norm2 = nn.GroupNorm(num_groups=8, num_channels=self.channels)
        self.conv2 = nn.Conv(self.channels, kernel_size=(3, 3), strides=(1, 1), padding=((1, 1), (1, 1)))

    def __call__(self, x):
        skip = x
        x = nn.elu(self.norm1(x))
        x = nn.elu(self.norm2(self.conv1(x)))
        return self.conv2(x) + skip

class DownBlock(nn.Module):
    channels_in: int
    channels_out: int

    def setup(self):
        self.bn1 = nn.GroupNorm(num_groups=8, num_channels=self.channels_in)
        self.conv1 = nn.Conv(self.channels_out, kernel_size=(3, 3), strides=(2, 2), padding=((1, 1), (1, 1)))
        self.bn2 = nn.GroupNorm(num_groups=8, num_channels=self.channels_out)
        self.conv2 = nn.Conv(self.channels_out, kernel_size=(3, 3), strides=(1, 1), padding=((1, 1), (1, 1)))
        self.conv3 = nn.Conv(self.channels_out, kernel_size=(3, 3), strides=(2, 2), padding=((1, 1), (1, 1)))

    def __call__(self, x):
        x_skip = self.conv3(nn.elu(self.bn1(x)))
        x = nn.elu(self.bn2(self.conv1(x)))
        return self.conv2(x) + x_skip

class UpBlock(nn.Module):
    channels_in: int
    channels_out: int

    def setup(self):
        self.bn1 = nn.GroupNorm(num_groups=8, num_channels=self.channels_in)
        self.conv1 = nn.Conv(self.channels_in, kernel_size=(3, 3), strides=(1, 1), padding=((1, 1), (1, 1)))
        self.bn2 = nn.GroupNorm(num_groups=8, num_channels=self.channels_in)
        self.conv2 = nn.Conv(self.channels_out, kernel_size=(3, 3), strides=(1, 1), padding=((1, 1), (1, 1)))
        self.conv3 = nn.Conv(self.channels_out, kernel_size=(3, 3), strides=(1, 1), padding=((1, 1), (1, 1)))

    def __call__(self, x):
        x = nn.upsample(x, scale=(2, 2), method="nearest")
        x_skip = self.conv3(x)
        x = nn.elu(self.bn2(self.conv1(x)))
        return self.conv2(x) + x_skip

class Encoder(nn.Module):
    channels: int
    ch: int = 32
    latent_channels: int = 32

    def setup(self):
        self.conv_1 = nn.Conv(self.ch, kernel_size=(3, 3), strides=(1, 1), padding=((1, 1), (1, 1)))
        self.conv_block1 = DownBlock(self.ch, self.ch * 2)
        self.conv_block2 = DownBlock(self.ch * 2, self.ch * 4)
        self.res_block_1 = ResBlock(self.ch * 4)
        self.res_block_2 = ResBlock(self.ch * 4)
        self.res_block_3 = ResBlock(self.ch * 4)
        self.conv_out = nn.Conv(self.ch * 4, kernel_size=(3, 3), strides=(1, 1), padding=((1, 1), (1, 1)))

    def __call__(self, x):
        x = self.conv_1(x)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.res_block_1(x)
        x = self.res_block_2(x)
        x = nn.elu(self.res_block_3(x))
        return self.conv_out(x)

class Decoder(nn.Module):
    channels: int
    ch: int = 32
    latent_channels: int = 32

    def setup(self):
        self.conv1 = nn.Conv(self.latent_channels, self.ch * 4, kernel_size=(3, 3), strides=(1, 1), padding=((1, 1), (1, 1)))
        self.res_block_1 = ResBlock(self.ch * 4)
        self.res_block_2 = ResBlock(self.ch * 4)
        self.conv_block1 = UpBlock(self.ch * 4, self.ch * 2)
        self.conv_block2 = UpBlock(self.ch * 2, self.ch)
        self.conv_out = nn.Conv(self.ch, self.channels, kernel_size=(3, 3), strides=(1, 1), padding=((1, 1), (1, 1)))

    def __call__(self, x):
        x = self.conv1(x)
        x = self.res_block_1(x)
        x = self.res_block_2(x)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        return nn.tanh(self.conv_out(x))
    
class VQVAE(nn.Module):
    channel_in: int
    ch: int = 16
    latent_channels: int = 32
    code_book_size: int = 64
    commitment_cost: float = 0.25

    def setup(self):
        self.encoder = Encoder(channels=self.channel_in, ch=self.ch, latent_channels=self.latent_channels)
        self.vq = VectorQuantizer(code_book_size=self.code_book_size, embedding_dim=self.latent_channels, commitment_cost=self.commitment_cost)
        self.decoder = Decoder(channels=self.channel_in, ch=self.ch, latent_channels=self.latent_channels)

    def encode(self, x):
        encoding = self.encoder(x)
        vq_loss, quantized, encoding_indices = self.vq(encoding)
        return vq_loss, quantized, encoding_indices

    def decode(self, x):
        return self.decoder(x)

    def __call__(self, x):
        vq_loss, quantized, encoding_indices = self.encode(x)
        recon = self.decode(quantized)
        return recon, vq_loss, quantized