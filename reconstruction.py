# import torch
# import matplotlib.pyplot as plt
# import torchvision.utils as vutils
# from model_imagenet import VQVAE
# from model_mnist import VQVAE
# from dataset import get_imagenet_dataloader, get_mnist_dataloader
# from config import imagenet_config, mnist_config

# # ✅ 저장된 체크포인트 불러오기
# def load_model(checkpoint_path):
#     model = VQVAE(
#         in_channels=3,
#         hidden_channels=128,
#         latent_dim=config["latent_dim"],
#         num_residual_layers=2,
#         residual_hidden_channels=32,
#         num_embeddings=config["num_embeddings"],
#         commitment_cost=config["commitment_cost"]
#     ).to(config["device"])

#     model.load_state_dict(torch.load(checkpoint_path, map_location=config["device"]))
#     model.eval()
#     print(f"✅ 모델 체크포인트 '{checkpoint_path}' 로드 완료!")
#     return model

# # ✅ 재구성된 이미지 출력 함수
# def show_reconstructed_images(model, data_loader, device, num_images=10, save_path=config["image_path"]):
#     model.eval()
#     original_images = []
#     reconstructed_images = []

#     with torch.no_grad():
#         for images, _ in data_loader:
#             images = images.to(device)
#             reconstructed, _ = model(images)
#             original_images.append(images.cpu())
#             reconstructed_images.append(reconstructed.cpu())
#             if len(original_images) * images.size(0) >= num_images:
#                 break

#     original_images = torch.cat(original_images)[:num_images]
#     reconstructed_images = torch.cat(reconstructed_images)[:num_images]

#     # ✅ 이미지 역정규화 (ImageNet Mean & Std)
#     def denormalize(tensor):
#         mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
#         std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
#         tensor = tensor * std + mean
#         return torch.clamp(tensor, 0, 1)

#     original_images = denormalize(original_images)
#     reconstructed_images = denormalize(reconstructed_images)

#     # ✅ 원본 vs 재구성 이미지 비교
#     fig, axes = plt.subplots(nrows=num_images, ncols=2, figsize=(10, num_images * 5))
#     for i in range(num_images):
#         axes[i, 0].imshow(original_images[i].permute(1, 2, 0))
#         axes[i, 0].set_title("Original Image")
#         axes[i, 0].axis("off")
#         axes[i, 1].imshow(reconstructed_images[i].permute(1, 2, 0))
#         axes[i, 1].set_title("Reconstructed Image")
#         axes[i, 1].axis("off")

#     plt.tight_layout()
#     plt.savefig(save_path)
#     print(f"✅ 재구성된 이미지가 '{save_path}'에 저장되었습니다.")
#     plt.close()

# # ✅ 실행 부분
# if __name__ == "__main__":
#     print("🔄 모델 로드 중...")
#     vqvae = load_model(config["checkpoint_path"])
    
#     print("📥 데이터 로드 중...")
#     _, test_loader = get_imagenet_dataloader()

#     print("🎨 이미지 재구성 중...")
#     show_reconstructed_images(vqvae, test_loader, config["device"], num_images=10)

# ##################################################################################

# checkpoint_path = config["checkpoint_path"]

# vqvae = VQVAE(
#     channel_in=1,
#     ch=16,
#     latent_channels=config["latent_dim"],
#     code_book_size=config["num_embeddings"],
#     commitment_cost=config["commitment_cost"]
# ).to(config["device"])

# vqvae.load_state_dict(torch.load(checkpoint_path, map_location=config["device"]))
# vqvae.eval()
# print(f"✅ 모델 체크포인트 '{checkpoint_path}' 로드 완료!")

# # ✅ 테스트 데이터 로드
# _, test_loader = get_mnist_dataloader()

# # ✅ 이미지 복원 함수
# def show_reconstructed_images(model, data_loader, device, num_images=10, save_path=config["image_path"]):
#     model.eval()
#     original_images = []
#     reconstructed_images = []

#     with torch.no_grad():
#         for images, _ in data_loader:
#             images = images.to(device)
#             reconstructed, _, _ = model(images)
#             # recon_data, vq_loss, quantized = model(images)
#             # vq_loss, quantized, encoding_indices = model.encode(images)
#             # encoding_indices[0]
            
#             original_images.append(images.cpu())
#             reconstructed_images.append(reconstructed.cpu())
#             if len(original_images) * images.size(0) >= num_images:
#                 break

#     original_images = torch.cat(original_images)[:num_images]
#     reconstructed_images = torch.cat(reconstructed_images)[:num_images]

#     # ✅ 원본 및 복원 이미지 시각화
#     fig, axes = plt.subplots(nrows=num_images, ncols=2, figsize=(10, num_images * 2))
#     for i in range(num_images):
#         axes[i, 0].imshow(original_images[i].squeeze(), cmap="gray")
#         axes[i, 0].set_title("Original")
#         axes[i, 0].axis("off")
#         axes[i, 1].imshow(reconstructed_images[i].squeeze(), cmap="gray")
#         axes[i, 1].set_title("Reconstructed")
#         axes[i, 1].axis("off")

#     plt.tight_layout()
#     plt.savefig(save_path)
#     print(f"✅ 재구성된 이미지가 '{save_path}'에 저장되었습니다.")
#     plt.close()

# # ✅ 복원 실행
# show_reconstructed_images(vqvae, test_loader, config["device"])


import torch
import matplotlib.pyplot as plt
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
    ).to(config["device"])
    model.load_state_dict(torch.load(checkpoint_path, map_location=config["device"]))
    model.eval()
    print(f"✅ 모델 체크포인트 '{checkpoint_path}' 로드 완료!")
    return model

def show_reconstructed_images(model, data_loader, device, num_images=10, save_path="output.png", is_mnist=False):
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

    fig, axes = plt.subplots(nrows=num_images, ncols=2, figsize=(10, num_images * 2))
    for i in range(num_images):
        if is_mnist:
            axes[i, 0].imshow(original_images[i].squeeze(), cmap="gray")
            axes[i, 1].imshow(reconstructed_images[i].squeeze(), cmap="gray")
        else:
            axes[i, 0].imshow(original_images[i].permute(1, 2, 0))
            axes[i, 1].imshow(reconstructed_images[i].permute(1, 2, 0))
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
    parser = argparse.ArgumentParser(description="VQ-VAE Reconstruction")
    parser.add_argument("--dataset", type=str, choices=["imagenet", "mnist"], required=True, help="Choose dataset: imagenet or mnist")
    args = parser.parse_args()

    if args.dataset == "imagenet":
        print("🔄 모델 로드 중...")
        vqvae = load_model(imagenet_config["checkpoint_path"], imagenet_config, VQVAE_Imagenet)
        print("📥 데이터 로드 중...")
        _, test_loader = get_imagenet_dataloader()
        print("🎨 이미지 재구성 중...")
        show_reconstructed_images(vqvae, test_loader, imagenet_config["device"], num_images=10, save_path=imagenet_config["image_path"], is_mnist=False)
    else:
        print("🔄 모델 로드 중...")
        vqvae = load_model(mnist_config["checkpoint_path"], mnist_config, VQVAE_MNIST)
        print("📥 데이터 로드 중...")
        _, test_loader = get_mnist_dataloader()
        print("🎨 이미지 재구성 중...")
        show_reconstructed_images(vqvae, test_loader, mnist_config["device"], num_images=10, save_path=mnist_config["image_path"], is_mnist=True)
