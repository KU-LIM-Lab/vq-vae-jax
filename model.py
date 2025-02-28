from flax import nnx
import jax
import jax.numpy as jnp


class ResidualBlock(nnx.Module):
    def __init__(self, hidden_dim, rngs: nnx.Rngs):
        self.hidden_dim = hidden_dim
        
        self.conv1 = nnx.Conv(
            in_features=hidden_dim, 
            out_features=hidden_dim, 
            kernel_size=(3, 3),  
            strides=(1, 1), 
            padding="SAME",
            rngs=rngs
            )
        self.conv2 = nnx.Conv(
            in_features=hidden_dim, 
            out_features=hidden_dim, 
            kernel_size=(1, 1),  
            strides=(1, 1), 
            padding="SAME",
            rngs=rngs
            )

    def __call__(self, x):
        residual = x
        x = nnx.relu(x)
        x = self.conv1(x)
        x = nnx.relu(x)
        x = self.conv2(x)
        return x + residual

class Encoder(nnx.Module):
    def __init__(self, in_channel, hidden_dim, out_channel, rngs: nnx.Rngs):
        self.conv1 = nnx.Conv(
            in_features=in_channel, 
            out_features=hidden_dim, 
            kernel_size=(4, 4), 
            strides=(2, 2),
            padding="SAME",
            rngs=rngs
            )
        self.conv2 = nnx.Conv(
            in_features=hidden_dim, 
            out_features=hidden_dim, 
            kernel_size=(4, 4), 
            strides=(2, 2),
            padding="SAME",
            rngs=rngs
            )

        self.residual1 = ResidualBlock(hidden_dim, rngs)
        self.residual2 = ResidualBlock(hidden_dim, rngs)
        
        self.conv3 = nnx.Conv(
            in_features=hidden_dim,
            out_features=out_channel,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="SAME",
            rngs=rngs
        )

    def __call__(self, x):
        x = self.conv1(x)
        x = nnx.relu(x)
        x = self.conv2(x)
        x = nnx.relu(x)
        x = self.residual1(x)
        x = self.residual2(x)
        x = self.conv3(x)
        return x

class Decoder(nnx.Module):
    def __init__(self, in_channel, hidden_dim, out_channel, rngs: nnx.Rngs):
        

        self.residual1 = ResidualBlock(in_channel, rngs)
        self.residual2 = ResidualBlock(in_channel, rngs)
        
        self.conv1 = nnx.ConvTranspose(
            in_features=in_channel, 
            out_features=hidden_dim, 
            kernel_size=(4, 4), 
            strides=(2, 2), 
            rngs=rngs
            )
        self.conv2 = nnx.ConvTranspose(
            in_features=hidden_dim, 
            out_features=hidden_dim, 
            kernel_size=(4, 4), 
            strides=(2, 2), 
            rngs=rngs
            )
        
        self.conv3 = nnx.Conv(
            in_features=hidden_dim,
            out_features=out_channel,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="SAME",
            rngs=rngs
        )

    def __call__(self, x):
        x = self.residual1(x)
        x = self.residual2(x)
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = nnx.relu(x)

        x = self.conv3(x)

        return x


class VeectorQuantizer(nnx.Module):
    def __init__(self, K, hidden_dim, beta, rngs: nnx.Rngs):
        key = rngs.params()
        
        self.K = K  # size of embedding space
        self.hidden_dim = hidden_dim  # dimension of embedding space
        self.beta = beta  # coefficient for commitment loss 
        
        # initialize embedding space
        self.embedding = nnx.Param(jax.random.uniform(key, (K, hidden_dim)))  
    
    def __call__(self, z_e):
        z_e_flattened = z_e.reshape(-1, self.hidden_dim)  # [B*H*W, D]
        
        # euclidean distance between z and embedding vectors
        distance = distance = jnp.sum(z_e_flattened**2, axis=1, keepdims=True) \
                                + jnp.sum(self.embedding**2, axis=1) \
                                - 2 * jnp.dot(z_e_flattened, self.embedding.T)
        # get nearest embedding
        embedding_indicies = jnp.argmin(distance, axis=1)
        embedding = jax.nn.one_hot(embedding_indicies, num_classes=self.K)
        z_q = jnp.dot(embedding, self.embedding.value).reshape(z_e.shape)
        
        # compute loss for embedding
        vq_loss = jnp.mean((jax.lax.stop_gradient(z_q) - z_e)**2)  # |sg[z_e] - e|^2_2
        commitment_loss = jnp.mean((z_e - jax.lax.stop_gradient(z_q))**2)  # |z_e - sg[e]|^2_2
        loss = vq_loss + self.beta * commitment_loss
        
        z_q = z_e + jax.lax.stop_gradient(z_q - z_e)  # z + (z_q - z).detach() in PyTorch
        
        return z_q, loss


class VQVAE(nnx.Module):
    def __init__(self, in_channel, hidden_dim, K, embedding_dim, beta, rngs: nnx.Rngs):
        self.encoder = Encoder(in_channel, hidden_dim, embedding_dim, rngs)
        self.vector_quantizer = VeectorQuantizer(K, embedding_dim, beta, rngs)
        self.decoder = Decoder(embedding_dim, hidden_dim, in_channel, rngs)
        
    def __call__(self, x):
        z_e = self.encoder(x)
        z_q, loss = self.vector_quantizer(z_e)  # z_e: encoder output, z_q: quantized output
        x_recon = self.decoder(z_q)
        return x_recon, loss


if __name__ == "__main__":
    key = jax.random.PRNGKey(0)

    x = jnp.ones((4, 32, 32, 3))
    
    encoder = Encoder(in_channel=3, hidden_dim=256, out_channel=10, rngs=nnx.Rngs(key))
    vector_quantizer = VeectorQuantizer(512, 256, 0.25, nnx.Rngs(key))
    decoder = Decoder(in_channel=10, hidden_dim=256, out_channel=3, rngs=nnx.Rngs(key))
    
    z = encoder(x)
    print("Encoded shape:", z.shape)
    
    z_q, loss = vector_quantizer(z)
    print("Quantized shape:", z_q.shape)
    
    reconstructed = decoder(z_q)
    print("Reconstructed shape:", reconstructed.shape)
    
    
    vqvae = VQVAE(in_channel=3, hidden_dim=256, K=512, embedding_dim=10, beta=0.25, rngs=nnx.Rngs(key))
    x_recon, loss = vqvae(x)
    print("Reconstructed shape:", x_recon.shape)
    
    reconstructed_loss = jnp.mean((x - x_recon)**2)
    print("Reconstruction loss:", reconstructed_loss)