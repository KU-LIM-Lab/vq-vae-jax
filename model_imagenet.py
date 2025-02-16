import torch
from torch import nn
from torch.nn import functional as F

from abc import ABC, abstractmethod
from torch import tensor as Tensor
from typing import List, Callable, Union, Any, TypeVar, Tuple


# class VAE(nn.Module, ABC):
#     def __init__(self) -> None:
#         super(VAE, self).__init__()

#     @abstractmethod
#     def encode(self, input: Tensor) -> List[Tensor]:
#         raise NotImplementedError

#     @abstractmethod
#     def decode(self, input: Tensor) -> Any:
#         raise NotImplementedError

#     def sample(self, num_samples: int, current_device: Union[int, str], **kwargs) -> Tensor:
#         raise NotImplementedError

#     def generate(self, x: Tensor, **kwargs) -> Tensor:
#         raise NotImplementedError
    
#     @abstractmethod
#     def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
#         pass

#     @abstractmethod
#     def loss_function(self, *args, **kwargs) -> dict:
#         pass


class VectorQuantizer(nn.Module):
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 beta: float): # beta는 논문 어디에?
        super(VectorQuantizer, self).__init__() # super.__init__과의 차이?
        self.K = num_embeddings
        self.D = embedding_dim
        self.beta = beta # commitment loss weight
        self.embedding = nn.Embedding(self.K, self.D) # embedding space(codebook) E
        self.embedding.weight.data.uniform_(-1 / self.K, 1 / self.K) # initialization, umiform prior distribution
        
    def forward(self, latents):
        latents = latents.permute(0, 2, 3, 1).contiguous()  # [B x D x H x W] -> [B x H x W x D]
        latents_shape = latents.shape
        flat_latents = latents.view(-1, self.D)  # [BHW x D]
        
        # Euclidean distance between latent vectors and embedding weights(encoding)
        dist = torch.sum(flat_latents ** 2, dim=1, keepdim=True) + \
               torch.sum(self.embedding.weight ** 2, dim=1) - \
               2 * torch.matmul(flat_latents, self.embedding.weight.t())  # [BHW x K]
        
        # Argmin
        encoding_index = torch.argmin(dist, dim=1).unsqueeze(1)  # [BHW, 1]
        
        # One-hot encodings
        device = latents.device
        encoding_one_hot = torch.zeros(encoding_index.size(0), self.K, device=device)
        encoding_one_hot.scatter_(1, encoding_index, 1)  # [BHW x K]
        
        # Quantize the latents
        quantized_latents = torch.matmul(encoding_one_hot, self.embedding.weight)  # [BHW, D]
        quantized_latents = quantized_latents.view(latents_shape)  # [B x H x W x D]

        # Compute the VQ Losses
        embedding_loss = F.mse_loss(quantized_latents, latents.detach()) # e: quantized_latents, Enc: latents
        commitment_loss = F.mse_loss(quantized_latents.detach(), latents) # detach: stop gradient

        vq_loss = embedding_loss + self.beta * commitment_loss

        # Add the residue back to the latents
        quantized_latents = latents + (quantized_latents - latents).detach()

        return vq_loss, quantized_latents.permute(0, 3, 1, 2).contiguous() # [B x D x H x W]


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=False),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1, stride=1, bias=False)
        )

    def forward(self, x):
        return x + self.block(x)


class ResidualStack(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, num_residual_layers):
        super(ResidualStack, self).__init__()
        self.layers = nn.ModuleList([ResidualBlock(in_channels, out_channels, hidden_channels) for _ in range(num_residual_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return F.relu(x)


class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, latent_dim, num_residual_layers, residual_hidden_channels):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=4, stride=2, padding=1)
        self.residual_stack = ResidualStack(hidden_channels, hidden_channels, residual_hidden_channels, num_residual_layers)
        self.conv3 = nn.Conv2d(hidden_channels, latent_dim, kernel_size=1, stride=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.residual_stack(x)
        x = self.conv3(x)
        return x


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_channels, out_channels, num_residual_layers, residual_hidden_channels):
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv2d(latent_dim, hidden_channels, kernel_size=3, stride=1, padding=1)
        self.residual_stack = ResidualStack(hidden_channels, hidden_channels, residual_hidden_channels, num_residual_layers)
        self.conv2 = nn.ConvTranspose2d(hidden_channels, hidden_channels // 2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.ConvTranspose2d(hidden_channels // 2, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.residual_stack(x)
        x = F.relu(self.conv2(x))
        x = torch.tanh(self.conv3(x))
        return x


class VQVAE(nn.Module):
    def __init__(self, in_channels, hidden_channels, latent_dim, num_residual_layers, residual_hidden_channels, num_embeddings, commitment_cost):
        super(VQVAE, self).__init__()
        self.encoder = Encoder(in_channels, hidden_channels, latent_dim, num_residual_layers, residual_hidden_channels)
        self.vector_quantizer = VectorQuantizer(num_embeddings, latent_dim, commitment_cost)
        self.decoder = Decoder(latent_dim, hidden_channels, in_channels, num_residual_layers, residual_hidden_channels)

    def forward(self, x):
        encoded = self.encoder(x)
        vq_loss, quantized = self.vector_quantizer(encoded)
        reconstructed = self.decoder(quantized)
        return reconstructed, vq_loss