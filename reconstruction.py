import torch
import matplotlib.pyplot as plt
import argparse
from torchvision.utils import make_grid

from model_cifar10 import VQVAE
from model_imagenet import VQVAE as VQVAE_Imagenet
from model_mnist import VQVAE as VQVAE_MNIST
from dataset import get_imagenet_dataloader, get_cifar10_dataloader, get_mnist_dataloader
from config import imagenet_config, cifar10_config, mnist_config


def load_model(checkpoint_path, config, model_class):    
    model = model_class(
        in_channels=config["in_channels"],
        hidden_channels=config["hidden_channels"], 
        latent_dim=config["latent_dim"],
        num_residual_layers=config["num_residual_layers"],
        residual_hidden_channels=config["residual_hidden_channels"],
        num_embeddings=config["num_embeddings"],
        commitment_cost=config["commitment_cost"]
    ).to(config["device"])
    
    model.load_state_dict(torch.load(checkpoint_path, map_location=config["device"]))
    model.eval()
    print(f"모델 체크포인트 '{checkpoint_path}' 로드 완료!")
    return model


def save_images_as_grid(images, save_path, is_mnist=False):
    grid = make_grid(images, nrow=5, padding=2, normalize=True)  # 한 줄에 5개씩 배치
    plt.figure(figsize=(10, 4))
    plt.imshow(grid.permute(1, 2, 0) if not is_mnist else grid.squeeze(), cmap="gray" if is_mnist else None)
    plt.axis("off")
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    print(f"✅ 이미지 저장 완료: {save_path}")
    plt.close()


def show_reconstructed_images(model, data_loader, device, num_images=10, original_path="originals.png", recon_path="reconstructions.png", is_mnist=False):
    model.eval()
    original_images, reconstructed_images = [], []

    with torch.no_grad():
        for images, _ in data_loader:
            images = images.to(device)
            reconstructed, *_ = model(images)
            original_images.append(images.cpu())
            reconstructed_images.append(reconstructed.cpu())
            if len(original_images) * images.size(0) >= num_images:
                break

    original_images = torch.cat(original_images)[:num_images]
    reconstructed_images = torch.cat(reconstructed_images)[:num_images]

    # 원본 이미지 저장
    save_images_as_grid(original_images, original_path, is_mnist)

    # 재구성 이미지 저장
    save_images_as_grid(reconstructed_images, recon_path, is_mnist)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VQ-VAE Reconstruction")
    
    parser.add_argument("--dataset", type=str, choices=["imagenet", "cifar10", "mnist"], required=True, help="Choose dataset: imagenet, cifar10, or mnist")
    
    args = parser.parse_args()

    if args.dataset == "cifar10":
        print("모델 로드 중...")
        vqvae = load_model(cifar10_config["checkpoint_path"], cifar10_config, VQVAE)
        print("데이터 로드 중...")
        _, test_loader = get_cifar10_dataloader()
        print("이미지 재구성 중...")
        show_reconstructed_images(vqvae, test_loader, cifar10_config["device"], num_images=10,
                                  original_path="/root/limlab/yeongyu/vqvae/experiments/originals.png", recon_path=cifar10_config["image_path"], is_mnist=False)

    elif args.dataset == "imagenet":
        print("모델 로드 중...")
        vqvae = load_model(imagenet_config["checkpoint_path"], imagenet_config, VQVAE_Imagenet)
        print("데이터 로드 중...")
        _, test_loader = get_imagenet_dataloader()
        print("이미지 재구성 중...")
        show_reconstructed_images(vqvae, test_loader, imagenet_config["device"], num_images=10, save_path=imagenet_config["image_path"], is_mnist=False)

    else:  # MNIST
        print("모델 로드 중...")
        vqvae = load_model(mnist_config["checkpoint_path"], mnist_config, VQVAE_MNIST)
        print("데이터 로드 중...")
        _, test_loader = get_mnist_dataloader()
        print("이미지 재구성 중...")
        show_reconstructed_images(vqvae, test_loader, mnist_config["device"], num_images=10, save_path=mnist_config["image_path"], is_mnist=True)
