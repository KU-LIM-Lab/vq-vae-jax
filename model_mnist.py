import torch
from torch import nn
from torch.nn import functional as F

from abc import ABC, abstractmethod
from torch import tensor as Tensor
from typing import List, Callable, Union, Any, TypeVar, Tuple


class VectorQuantizer(nn.Module):
    def __init__(self, code_book_size, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        self.code_book_size = code_book_size
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        self.embedding = nn.Embedding(code_book_size, embedding_dim)
        self.embedding.weight.data.uniform_(-1/code_book_size, 1/code_book_size)

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 3, 1).contiguous()  # BSxCxHxW --> BSxHxWxC
        input_shape = inputs.shape
        
        flat_input = inputs.view(-1, 1, self.embedding_dim)  # BSxHxWxC --> BS*H*Wx1xC
        
        # Calculate the distance between each embedding and each codebook vector
        distances = (flat_input - self.embedding.weight.unsqueeze(0)).pow(2).mean(2)  # BS*H*WxN
        
        # Find the closest codebook vector
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)  # BS*H*Wx1
        
        # Select that codebook vector
        quantized = self.embedding(encoding_indices).view(input_shape)
        
        # Create loss that pulls encoder embeddings and codebook vector selected
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        # Reconstruct quantized representation using the encoder embeddings to allow for 
        # backpropagation of gradients into encoder
        if self.training:
            quantized = inputs + (quantized - inputs).detach()
        
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), encoding_indices.reshape(input_shape[0], -1)


class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.norm2 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        
    def forward(self, x):
        skip = x
        
        x = F.elu(self.norm1(x))
        x = F.elu(self.norm2(self.conv1(x)))
        x = self.conv2(x) + skip
        return x


# We split up our network into two parts, the Encoder and the Decoder
class DownBlock(nn.Module):
    def __init__(self, channels_in, channels_out):
        super(DownBlock, self).__init__()
        self.bn1 = nn.GroupNorm(8, channels_in)
        self.conv1 = nn.Conv2d(channels_in, channels_out, 3, 2, 1)
        self.bn2 = nn.GroupNorm(8, channels_out)
        self.conv2 = nn.Conv2d(channels_out, channels_out, 3, 1, 1)
        
        self.conv3 = nn.Conv2d(channels_in, channels_out, 3, 2, 1)

    def forward(self, x):
        x = F.elu(self.bn1(x))
                  
        x_skip = self.conv3(x)
        
        x = F.elu(self.bn2(self.conv1(x)))        
        return self.conv2(x) + x_skip
    
    
# We split up our network into two parts, the Encoder and the Decoder
class UpBlock(nn.Module):
    def __init__(self, channels_in, channels_out):
        super(UpBlock, self).__init__()
        self.bn1 = nn.GroupNorm(8, channels_in)

        self.conv1 = nn.Conv2d(channels_in, channels_in, 3, 1, 1)
        self.bn2 = nn.GroupNorm(8, channels_in)

        self.conv2 = nn.Conv2d(channels_in, channels_out, 3, 1, 1)
        
        self.conv3 = nn.Conv2d(channels_in, channels_out, 3, 1, 1)
        self.up_nn = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x_in):
        x = self.up_nn(F.elu(self.bn1(x_in)))
        
        x_skip = self.conv3(x)
        
        x = F.elu(self.bn2(self.conv1(x)))
        return self.conv2(x) + x_skip

    
# We split up our network into two parts, the Encoder and the Decoder
class Encoder(nn.Module):
    def __init__(self, channels, ch=32, latent_channels=32):
        super(Encoder, self).__init__()
        self.conv_1 = nn.Conv2d(channels, ch, 3, 1, 1)
        
        self.conv_block1 = DownBlock(ch, ch * 2)
        self.conv_block2 = DownBlock(ch * 2, ch * 4)

        # Instead of flattening (and then having to unflatten) out our feature map and 
        # putting it through a linear layer we can just use a conv layer
        # where the kernal is the same size as the feature map 
        # (in practice it's the same thing)
        self.res_block_1 = ResBlock(ch * 4)
        self.res_block_2 = ResBlock(ch * 4)
        self.res_block_3 = ResBlock(ch * 4)

        self.conv_out = nn.Conv2d(4 * ch, latent_channels, 3, 1, 1)
    
    def forward(self, x):
        x = self.conv_1(x)
        
        x = self.conv_block1(x)
        x = self.conv_block2(x)

        x = self.res_block_1(x)
        x = self.res_block_2(x)
        x = F.elu(self.res_block_3(x))

        return self.conv_out(x)
    
    
class Decoder(nn.Module):
    def __init__(self, channels, ch = 32, latent_channels = 32):
        super(Decoder, self).__init__()
        
        self.conv1 = nn.Conv2d(latent_channels, 4 * ch, 3, 1, 1)
        self.res_block_1 = ResBlock(ch * 4)
        self.res_block_2 = ResBlock(ch * 4)
        self.res_block_2 = ResBlock(ch * 4)

        self.conv_block1 = UpBlock(4 * ch, 2 * ch)
        self.conv_block2 = UpBlock(2 * ch, ch)
        self.conv_out = nn.Conv2d(ch, channels, 3, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.res_block_1(x)
        x = self.res_block_2(x)
        x = self.res_block_2(x)

        x = self.conv_block1(x)
        x = self.conv_block2(x)
        
        return torch.tanh(self.conv_out(x))


class VQVAE(nn.Module):
    def __init__(self, channel_in, ch=16, latent_channels=32, code_book_size=64, commitment_cost=0.25):
        super(VQVAE, self).__init__()
        self.encoder = Encoder(channels=channel_in, ch=ch, latent_channels=latent_channels)
        
        self.vq = VectorQuantizer(code_book_size=code_book_size, 
                                  embedding_dim=latent_channels, 
                                  commitment_cost=commitment_cost)
        
        self.decoder = Decoder(channels=channel_in, ch=ch, latent_channels=latent_channels)

    def encode(self, x):
        encoding = self.encoder(x)
        vq_loss, quantized, encoding_indices = self.vq(encoding)
        return vq_loss, quantized, encoding_indices
        
    def decode(self, x):
        return self.decoder(x)
        
    def forward(self, x):
        vq_loss, quantized, encoding_indices = self.encode(x)
        recon = self.decode(quantized)
        
        return recon, vq_loss, quantized
    

# class ResidualLayer(nn.Module):

#     def __init__(self,
#                  in_channels: int,
#                  out_channels: int):
#         super(ResidualLayer, self).__init__()
#         self.resblock = nn.Sequential(nn.Conv2d(in_channels, out_channels,
#                                                 kernel_size=3, padding=1, bias=False),
#                                       nn.ReLU(True),
#                                       nn.Conv2d(out_channels, out_channels,
#                                                 kernel_size=1, bias=False))

#     def forward(self, input: Tensor) -> Tensor:
#         return input + self.resblock(input)


# class VQVAE(VAE):
#     def __init__(self,
#                  in_channels: int,
#                  embedding_dim: int,
#                  num_embeddings: int,
#                  hidden_dims: List = None,
#                  beta: float = 0.25,
#                  img_size: int = 64,
#                  **kwargs) -> None:
#         super(VQVAE, self).__init__()

#         self.embedding_dim = embedding_dim
#         self.num_embeddings = num_embeddings
#         self.img_size = img_size
#         self.beta = beta

#         modules = []
#         if hidden_dims is None:
#             hidden_dims = [128, 256]

#         # Build Encoder
#         for h_dim in hidden_dims:
#             modules.append(
#                 nn.Sequential(
#                     nn.Conv2d(in_channels, out_channels=h_dim,
#                               kernel_size=4, stride=2, padding=1),
#                     nn.LeakyReLU())
#             )
#             in_channels = h_dim

#         modules.append(
#             nn.Sequential(
#                 nn.Conv2d(in_channels, in_channels,
#                           kernel_size=3, stride=1, padding=1),
#                 nn.LeakyReLU())
#         )

#         for _ in range(6):
#             modules.append(ResidualLayer(in_channels, in_channels))
#         modules.append(nn.LeakyReLU())

#         modules.append(
#             nn.Sequential(
#                 nn.Conv2d(in_channels, embedding_dim,
#                           kernel_size=1, stride=1),
#                 nn.LeakyReLU())
#         )

#         self.encoder = nn.Sequential(*modules)

#         self.vq_layer = VectorQuantizer(num_embeddings,
#                                         embedding_dim,
#                                         self.beta)

#         # Build Decoder
#         modules = []
#         modules.append(
#             nn.Sequential(
#                 nn.Conv2d(embedding_dim,
#                           hidden_dims[-1],
#                           kernel_size=3,
#                           stride=1,
#                           padding=1),
#                 nn.LeakyReLU())
#         )

#         for _ in range(6):
#             modules.append(ResidualLayer(hidden_dims[-1], hidden_dims[-1]))

#         modules.append(nn.LeakyReLU())

#         hidden_dims.reverse()

#         for i in range(len(hidden_dims) - 1):
#             modules.append(
#                 nn.Sequential(
#                     nn.ConvTranspose2d(hidden_dims[i],
#                                        hidden_dims[i + 1],
#                                        kernel_size=4,
#                                        stride=2,
#                                        padding=1),
#                     nn.LeakyReLU())
#             )

#         modules.append(
#             nn.Sequential(
#                 nn.ConvTranspose2d(hidden_dims[-1],
#                                    out_channels=3,
#                                    kernel_size=4,
#                                    stride=2, padding=1),
#                 nn.Tanh()))

#         self.decoder = nn.Sequential(*modules)

#     def encode(self, input: Tensor) -> List[Tensor]:
#         result = self.encoder(input) # Defined in above block
#         return [result]

#     def decode(self, z: Tensor) -> Tensor:
#         result = self.decoder(z)
#         return result

#     def loss_function(self,
#                       *args,
#                       **kwargs) -> dict:
#         recons = args[0] # recons?
#         input = args[1]
#         vq_loss = args[2]

#         recons_loss = F.mse_loss(recons, input) # ELBO

#         loss = recons_loss + vq_loss
#         return {'loss': loss,
#                 'Reconstruction_Loss': recons_loss,
#                 'VQ_Loss':vq_loss}
    
#     def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
#         encoding = self.encode(input)[0] # Why first index?
#         quantized_inputs, vq_loss = self.vq_layer(encoding)
#         return [self.decode(quantized_inputs), input, vq_loss]

#     def sample(self,
#                num_samples: int,
#                current_device: Union[int, str], **kwargs) -> Tensor:
#     #     raise Warning('VQVAE sampler is not implemented.')
#         z = torch.randint(0, self.num_embeddings, (num_samples, self.img_size // 4, self.img_size // 4)).to(current_device)
#         quantized = self.vq_layer.embedding(z)  # [B, H, W, embedding_dim]
#         quantized = quantized.permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]
#         samples = self.decode(quantized)
#         return samples  # [num_samples, 3, 64, 64]

#     def generate(self, x: Tensor, **kwargs) -> Tensor:
#         return self.forward(x)[0]
