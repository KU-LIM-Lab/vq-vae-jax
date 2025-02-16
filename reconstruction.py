import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import matplotlib.pyplot as plt
import pickle
from model_imagenet import VQVAE as VQVAE_Imagenet
from model_mnist import VQVAE as VQVAE_MNIST
from dataset import get_imagenet_dataloader, get_mnist_dataloader
from config import imagenet_config, mnist_config

def load_model(checkpoint_path, config, model_class):
    model = model_class(
        in_channels=config.get("channel_in", 3),
        hidden_channels=config.get("hidden_channels", 128),
        latent_dim=config["latent_dim"],
        num_residual_layers=config.get("num_residual_layers", 2),
        residual_hidden_channels=config.get("residual_hidden_channels", 32),
        num_embeddings=config["num_embeddings"],
        commitment_cost=config["commitment_cost"]
    )
    with open(checkpoint_path, "rb") as f:
        params = pickle.load(f)
    print(f"✅ 모델 체크포인트 '{checkpoint_path}' 로드 완료!")
    return model, params

def show_reconstructed_images(model, params, data_loader, num_images=10, save_path="output.png", is_mnist=False):
    original_images, reconstructed_images = [], []
    
    for batch in data_loader.take(num_images // batch.shape[0]):
        recon_images, _ = model.apply({"params": params}, batch)
        original_images.append(batch)
        reconstructed_images.append(recon_images)
    
    original_images = np.concatenate(original_images)[:num_images]
    reconstructed_images = np.concatenate(reconstructed_images)[:num_images]

    fig, axes = plt.subplots(nrows=num_images, ncols=2, figsize=(10, num_images * 2))
    for i in range(num_images):
        if is_mnist:
            axes[i, 0].imshow(original_images[i].squeeze(), cmap="gray")
            axes[i, 1].imshow(reconstructed_images[i].squeeze(), cmap="gray")
        else:
            axes[i, 0].imshow(original_images[i].transpose(1, 2, 0))
            axes[i, 1].imshow(reconstructed_images[i].transpose(1, 2, 0))
        axes[i, 0].set_title("Original")
        axes[i, 0].axis("off")
        axes[i, 1].set_title("Reconstructed")
        axes[i, 1].axis("off")
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"✅ 재구성된 이미지가 '{save_path}'에 저장되었습니다.")
    plt.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="VQ-VAE Reconstruction (JAX & Flax)")
    parser.add_argument("--dataset", type=str, choices=["imagenet", "mnist"], required=True, help="Choose dataset: imagenet or mnist")
    args = parser.parse_args()

    if args.dataset == "imagenet":
        print("🔄 모델 로드 중...")
        vqvae, params = load_model(imagenet_config["checkpoint_path"], imagenet_config, VQVAE_Imagenet)
        print("📥 데이터 로드 중...")
        _, test_loader = get_imagenet_dataloader()
        print("🎨 이미지 재구성 중...")
        show_reconstructed_images(vqvae, params, test_loader, num_images=10, save_path=imagenet_config["image_path"], is_mnist=False)
    else:
        print("🔄 모델 로드 중...")
        vqvae, params = load_model(mnist_config["checkpoint_path"], mnist_config, VQVAE_MNIST)
        print("📥 데이터 로드 중...")
        _, test_loader = get_mnist_dataloader()
        print("🎨 이미지 재구성 중...")
        show_reconstructed_images(vqvae, params, test_loader, num_images=10, save_path=mnist_config["image_path"], is_mnist=True)
