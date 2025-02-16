from flax.core import FrozenDict
import jax

# Check available device (CPU, GPU, or TPU)
def get_device():
    devices = jax.devices()
    return "cuda" if any(d.device_kind == "NVIDIA GPU" for d in devices) else "cpu"

# Configuration for ImageNet mini
imagenet_config = FrozenDict({
    "batch_size": 128,
    "lr": 2e-4,
    "epochs": 100,
    "image_size": 128,
    "latent_dim": 32,
    "num_embeddings": 512,
    "commitment_cost": 0.25,
    "num_residual_layers": 2,
    "residual_hidden_channels": 32,
    "device": get_device(),
    "checkpoint_path": "/root/limlab/yeongyu/vqvae-jax/checkpoint/best_valid_loss_imagenet.pth",
    "image_path": "/root/limlab/yeongyu/vqvae-jax/experiments/reconstructed_images_imagenet.png"
})

# Configuration for MNIST
mnist_config = FrozenDict({
    "batch_size": 64,
    "lr": 1e-4,
    "epochs": 80,
    "image_size": 32,  # Resizing MNIST to 32x32
    "latent_dim": 10,  # Smaller latent dim for simpler dataset
    "num_embeddings": 32,
    "commitment_cost": 0.25,
    "num_residual_layers": 3,
    "residual_hidden_channels": 32,
    "device": get_device(),
    "checkpoint_path": "/root/limlab/yeongyu/vqvae/checkpoint/best_valid_loss_mnist.pth",
    "image_path": "/root/limlab/yeongyu/vqvae/experiments/reconstructed_images_mnist.png"
})
