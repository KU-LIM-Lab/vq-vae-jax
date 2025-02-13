import torch
from torch import nn
from torch.nn import functional as F

from abc import ABC, abstractmethod
from torch import tensor as Tensor
from typing import List, Callable, Union, Any, TypeVar, Tuple


class VAE(nn.Module, ABC):
    def __init__(self) -> None:
        super(VAE, self).__init__()

    @abstractmethod
    def encode(self, input: Tensor) -> List[Tensor]:
        raise NotImplementedError

    @abstractmethod
    def decode(self, input: Tensor) -> Any:
        raise NotImplementedError

    def sample(self, num_samples: int, current_device: Union[int, str], **kwargs) -> Tensor:
        raise NotImplementedError

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError
    
    @abstractmethod
    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        pass

    @abstractmethod
    def loss_function(self, *args, **kwargs) -> dict:
        pass


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

        return quantized_latents.permute(0, 3, 1, 2).contiguous(), vq_loss  # [B x D x H x W]


class ResidualLayer(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int):
        super(ResidualLayer, self).__init__()
        self.resblock = nn.Sequential(nn.Conv2d(in_channels, out_channels,
                                                kernel_size=3, padding=1, bias=False),
                                      nn.ReLU(True),
                                      nn.Conv2d(out_channels, out_channels,
                                                kernel_size=1, bias=False))

    def forward(self, input: Tensor) -> Tensor:
        return input + self.resblock(input)


class VQVAE(VAE):
    def __init__(self,
                 in_channels: int,
                 embedding_dim: int,
                 num_embeddings: int,
                 hidden_dims: List = None,
                 beta: float = 0.25,
                 img_size: int = 64,
                 **kwargs) -> None:
        super(VQVAE, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.img_size = img_size
        self.beta = beta

        modules = []
        if hidden_dims is None:
            hidden_dims = [128, 256]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=4, stride=2, padding=1),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels,
                          kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU())
        )

        for _ in range(6):
            modules.append(ResidualLayer(in_channels, in_channels))
        modules.append(nn.LeakyReLU())

        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, embedding_dim,
                          kernel_size=1, stride=1),
                nn.LeakyReLU())
        )

        self.encoder = nn.Sequential(*modules)

        self.vq_layer = VectorQuantizer(num_embeddings,
                                        embedding_dim,
                                        self.beta)

        # Build Decoder
        modules = []
        modules.append(
            nn.Sequential(
                nn.Conv2d(embedding_dim,
                          hidden_dims[-1],
                          kernel_size=3,
                          stride=1,
                          padding=1),
                nn.LeakyReLU())
        )

        for _ in range(6):
            modules.append(ResidualLayer(hidden_dims[-1], hidden_dims[-1]))

        modules.append(nn.LeakyReLU())

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=4,
                                       stride=2,
                                       padding=1),
                    nn.LeakyReLU())
            )

        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(hidden_dims[-1],
                                   out_channels=3,
                                   kernel_size=4,
                                   stride=2, padding=1),
                nn.Tanh()))

        self.decoder = nn.Sequential(*modules)

    def encode(self, input: Tensor) -> List[Tensor]:
        result = self.encoder(input) # Defined in above block
        return [result]

    def decode(self, z: Tensor) -> Tensor:
        result = self.decoder(z)
        return result

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        recons = args[0] # recons?
        input = args[1]
        vq_loss = args[2]

        recons_loss = F.mse_loss(recons, input) # ELBO

        loss = recons_loss + vq_loss
        return {'loss': loss,
                'Reconstruction_Loss': recons_loss,
                'VQ_Loss':vq_loss}
    
    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        encoding = self.encode(input)[0] # Why first index?
        quantized_inputs, vq_loss = self.vq_layer(encoding)
        return [self.decode(quantized_inputs), input, vq_loss]

    def sample(self,
               num_samples: int,
               current_device: Union[int, str], **kwargs) -> Tensor:
    #     raise Warning('VQVAE sampler is not implemented.')
        z = torch.randint(0, self.num_embeddings, (num_samples, self.img_size // 4, self.img_size // 4)).to(current_device)
        quantized = self.vq_layer.embedding(z)  # [B, H, W, embedding_dim]
        quantized = quantized.permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]
        samples = self.decode(quantized)
        return samples  # [num_samples, 3, 64, 64]

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        return self.forward(x)[0]
