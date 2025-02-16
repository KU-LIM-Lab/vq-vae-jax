import torch

# Configuration for ImageNet mini
imagenet_config = {
    "batch_size": 128, 
    "lr": 2e-4,  
    "epochs": 100,  
    "image_size": 128,  
    "latent_dim": 32,   
    "num_embeddings": 512,
    "commitment_cost": 0.25,
    "num_residual_layers": 2,
    "residual_hidden_channels": 32,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "checkpoint_path": "/root/limlab/yeongyu/vqvae/checkpoint/best_valid_loss.pth",
    "image_path": "/root/limlab/yeongyu/vqvae/experiments/reconstructed_images.png"
}

# Configuration for MNIST
mnist_config = {
    "batch_size": 64,
    "lr": 1e-4,
    "epochs": 80,
    "image_size": 32,  # MNIST 이미지를 32x32로 변환
    "latent_dim": 10,  # MNIST는 ImageNet보다 단순하므로 latent dim 감소
    "num_embeddings": 32,
    "commitment_cost": 0.25,
    "num_residual_layers": 3,
    "residual_hidden_channels": 32,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "checkpoint_path": "best_valid_loss.pth",
    "checkpoint_path": "/root/limlab/yeongyu/vqvae/checkpoint/best_valid_loss_mnist.pth",
    "image_path": "/root/limlab/yeongyu/vqvae/experiments/reconstructed_images_mnist.png"
}