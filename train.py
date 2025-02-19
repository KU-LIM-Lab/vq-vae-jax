import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import os
import wandb
from tqdm import tqdm
from torchvision.utils import save_image
from pytorch_fid.fid_score import calculate_fid_given_paths

from model_cifar10 import VQVAE
from model_imagenet import VQVAE as VQVAE_Imagenet
from model_mnist import VQVAE as VQVAE_MNIST
from dataset import get_imagenet_dataloader, get_cifar10_dataloader, get_mnist_dataloader
from config import imagenet_config, cifar10_config, mnist_config


def train_cifar10():
    wandb.init(project="vqvae-cifar10", config=cifar10_config)
    train_loader, test_loader = get_cifar10_dataloader()

    vqvae = VQVAE(
        in_channels=cifar10_config["in_channels"],
        hidden_channels=cifar10_config["hidden_channels"], # 256
        latent_dim=cifar10_config["latent_dim"], # 10
        num_residual_layers=cifar10_config["num_residual_layers"],
        residual_hidden_channels=cifar10_config["residual_hidden_channels"],
        num_embeddings=cifar10_config["num_embeddings"],
        commitment_cost=cifar10_config["commitment_cost"]
    ).to(cifar10_config["device"])
    
    optimizer = optim.Adam(vqvae.parameters(), lr=cifar10_config["lr"])
    best_valid_loss = float("inf")

    for epoch in range(cifar10_config["epochs"]):
        vqvae.train()
        total_train_loss, total_train_vq_loss = 0, 0

        for images, _ in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            images = images.to(cifar10_config["device"])
            optimizer.zero_grad()
            reconstructed, train_vq_loss = vqvae(images)
            train_loss = F.mse_loss(reconstructed, images) + train_vq_loss
            train_loss.backward()
            optimizer.step()

            total_train_loss += train_loss.item()
            total_train_vq_loss += train_vq_loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_train_vq_loss = total_train_vq_loss / len(train_loader)

        vqvae.eval()
        total_valid_loss, total_valid_vq_loss = 0, 0

        with torch.no_grad():
            for val_images, _ in tqdm(test_loader, desc="Validating"):
                val_images = val_images.to(cifar10_config["device"])
                recon_images, valid_vq_loss = vqvae(val_images)
                
                valid_loss = F.mse_loss(recon_images, val_images) + valid_vq_loss
                total_valid_loss += valid_loss.item()
                total_valid_vq_loss += valid_vq_loss.item()

        avg_valid_loss = total_valid_loss / len(test_loader)
        avg_valid_vq_loss = total_valid_vq_loss / len(test_loader)

        wandb.log({"epoch": epoch + 1, "train_loss": avg_train_loss, "train_vq_loss": avg_train_vq_loss, "valid_loss": avg_valid_loss, "valid_vq_loss": avg_valid_vq_loss})#, "fid_score": fid_score})

        print(f"Epoch [{epoch+1}/{cifar10_config['epochs']}], Train Loss: {avg_train_loss:.4f}, Valid Loss: {avg_valid_loss:.4f}")#, FID: {fid_score:.2f}")
        
        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            torch.save(vqvae.state_dict(), cifar10_config["checkpoint_path"])
            print(f"Checkpoint Saved: {cifar10_config['checkpoint_path']} (Valid Loss: {best_valid_loss:.4f})")


def fid_cifar10():
    """체크포인트 로드 후 validation 과정에서 FID Score 측정 (pytorch-fid 활용)"""

    _, test_loader = get_cifar10_dataloader()
    
    vqvae = VQVAE(
        in_channels=cifar10_config["in_channels"],
        hidden_channels=cifar10_config["hidden_channels"], # 256
        latent_dim=cifar10_config["latent_dim"], # 10
        num_residual_layers=cifar10_config["num_residual_layers"],
        residual_hidden_channels=cifar10_config["residual_hidden_channels"],
        num_embeddings=cifar10_config["num_embeddings"],
        commitment_cost=cifar10_config["commitment_cost"]
    ).to(cifar10_config["device"])

    checkpoint_path = cifar10_config["checkpoint_path"]
    if os.path.exists(checkpoint_path):
        print(f"기존 체크포인트 로드: {checkpoint_path}")
        vqvae.load_state_dict(torch.load(checkpoint_path, map_location=cifar10_config["device"]))
    else:
        print(f"체크포인트 {checkpoint_path} 가 존재하지 않습니다.")
        return
    
    vqvae.eval()

    real_images_path = "cifar10_real"
    fake_images_path = "cifar10_fake"
    os.makedirs(real_images_path, exist_ok=True)
    os.makedirs(fake_images_path, exist_ok=True)

    with torch.no_grad():
        for i, (val_images, _) in enumerate(tqdm(test_loader, desc="Saving Images for FID")):
            val_images = val_images.to(cifar10_config["device"])
            recon_images, _ = vqvae(val_images)

            # 이미지 저장 (실제 이미지 & 생성 이미지)
            for j in range(val_images.size(0)):
                save_image(val_images[j], f"{real_images_path}/{i * val_images.size(0) + j}.png")
                save_image(recon_images[j], f"{fake_images_path}/{i * val_images.size(0) + j}.png")

    print("이미지 저장 완료! FID 계산 시작")

    # FID Score 계산 (pytorch-fid 활용)
    fid_score = calculate_fid_given_paths(
        [real_images_path, fake_images_path],
        batch_size=50,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        dims=2048  # 기본적으로 Inception v3의 feature layer 사용
    )

    print(f"FID Score: {fid_score:.4f}")


# def train_imagenet():
#     wandb.init(project="vqvae-imagenet", config=imagenet_config)
#     train_loader, test_loader = get_imagenet_dataloader()
#     vqvae = VQVAE_Imagenet(3, 128, imagenet_config["latent_dim"], 2, 32, imagenet_config["num_embeddings"], imagenet_config["commitment_cost"]).to(imagenet_config["device"])

#     inception_model = inception_v3(pretrained=True, transform_input=False).to(imagenet_config["device"])
#     inception_model.fc = torch.nn.Identity()
#     inception_model.eval()

#     optimizer = optim.Adam(vqvae.parameters(), lr=imagenet_config["lr"])
#     best_valid_loss = float("inf")

#     for epoch in range(imagenet_config["epochs"]):
#         vqvae.train()
#         total_train_loss, total_train_vq_loss = 0, 0

#         for images, _ in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
#             images = images.to(imagenet_config["device"])
#             optimizer.zero_grad()
#             reconstructed, train_vq_loss = vqvae(images)
#             train_loss = F.mse_loss(reconstructed, images) + train_vq_loss
#             train_loss.backward()
#             optimizer.step()

#             total_train_loss += train_loss.item()
#             total_train_vq_loss += train_vq_loss.item()

#         avg_train_loss = total_train_loss / len(train_loader)
#         avg_train_vq_loss = total_train_vq_loss / len(train_loader)

#         vqvae.eval()
#         total_valid_loss, total_valid_vq_loss = 0, 0
#         real_features, fake_features = [], []

#         with torch.no_grad():
#             for val_images, _ in tqdm(test_loader, desc="Validating"):
#                 val_images = val_images.to(imagenet_config["device"])
#                 recon_images, valid_vq_loss = vqvae(val_images)
#                 valid_loss = F.mse_loss(recon_images, val_images) + valid_vq_loss

#                 total_valid_loss += valid_loss.item()
#                 total_valid_vq_loss += valid_vq_loss.item()

#                 real_features.append(get_features(val_images, inception_model, imagenet_config["device"]))
#                 fake_features.append(get_features(recon_images, inception_model, imagenet_config["device"]))

#         avg_valid_loss = total_valid_loss / len(test_loader)
#         avg_valid_vq_loss = total_valid_vq_loss / len(test_loader)
#         fid_score = calculate_fid(np.concatenate(real_features, axis=0), np.concatenate(fake_features, axis=0))

#         wandb.log({"epoch": epoch + 1, "train_loss": avg_train_loss, "train_vq_loss": avg_train_vq_loss, "valid_loss": avg_valid_loss, "valid_vq_loss": avg_valid_vq_loss, "fid_score": fid_score})

#         print(f"Epoch [{epoch+1}/{imagenet_config['epochs']}], Train Loss: {avg_train_loss:.4f}, Valid Loss: {avg_valid_loss:.4f}, FID: {fid_score:.2f}")
        
#         if avg_valid_loss < best_valid_loss:
#             best_valid_loss = avg_valid_loss
#             torch.save(vqvae.state_dict(), imagenet_config["checkpoint_path"])
#             print(f"✅ Checkpoint Saved: {imagenet_config['checkpoint_path']} (Valid Loss: {best_valid_loss:.4f})")


# def test_imagenet():
#     """체크포인트 로드 후 validation 과정에서 FID Score만 측정 (pytorch-fid 사용)"""
    
#     # 데이터 로더 불러오기 (검증 데이터셋만 사용)
#     _, test_loader = get_imagenet_dataloader()

#     # 모델 초기화 (VQ-VAE)
#     vqvae = VQVAE_Imagenet(
#         3, 128, imagenet_config["latent_dim"],
#         2, 32, imagenet_config["num_embeddings"], imagenet_config["commitment_cost"]
#     ).to(imagenet_config["device"])

#     # 🔹 체크포인트 로드
#     checkpoint_path = imagenet_config["checkpoint_path"]
#     if os.path.exists(checkpoint_path):
#         print(f"🔄 기존 체크포인트 로드: {checkpoint_path}")
#         vqvae.load_state_dict(torch.load(checkpoint_path, map_location=imagenet_config["device"]))
#     else:
#         print(f"❌ 체크포인트 {checkpoint_path} 가 존재하지 않습니다.")
#         return
    
#     vqvae.eval()

#     # 🔹 FID 계산을 위한 이미지 저장 폴더 설정
#     real_images_path = "imagenet_real"
#     fake_images_path = "imagenet_fake"
#     os.makedirs(real_images_path, exist_ok=True)
#     os.makedirs(fake_images_path, exist_ok=True)

#     # 🔹 Validation 과정에서 FID Score 측정
#     with torch.no_grad():
#         for i, (val_images, _) in enumerate(tqdm(test_loader, desc="Saving Images for FID")):
#             val_images = val_images.to(imagenet_config["device"])
#             recon_images, _ = vqvae(val_images)

#             # 🔹 이미지 저장 (실제 이미지 & 생성 이미지)
#             for j in range(val_images.size(0)):
#                 save_image(val_images[j], f"{real_images_path}/{i * val_images.size(0) + j}.png")
#                 save_image(recon_images[j], f"{fake_images_path}/{i * val_images.size(0) + j}.png")

#     print("✅ 이미지 저장 완료! FID 계산 시작")

#     # 🔹 FID Score 계산 (pytorch-fid 활용)
#     fid_score = calculate_fid_given_paths(
#         [real_images_path, fake_images_path],
#         batch_size=50,
#         device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
#         dims=2048  # 기본적으로 Inception v3의 feature layer 사용
#     )

#     print(f"✅ FID Score: {fid_score:.4f}")


# def train_mnist():
#     wandb.init(project="vqvae-mnist", config=mnist_config)
#     train_loader, test_loader = get_mnist_dataloader()
#     vqvae = VQVAE_MNIST(1, 16, mnist_config["latent_dim"], mnist_config["num_embeddings"], mnist_config["commitment_cost"]).to(mnist_config["device"])
#     resnet_model = ResNetFeatureExtractor().to(mnist_config["device"])
#     resnet_model.eval()

#     optimizer = optim.Adam(vqvae.parameters(), lr=mnist_config["lr"])
#     best_valid_loss = float("inf")

#     for epoch in range(mnist_config["epochs"]):
#         vqvae.train()
#         total_train_loss, total_train_vq_loss = 0, 0

#         for images, _ in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
#             images = images.to(mnist_config["device"])
#             optimizer.zero_grad()
#             reconstructed, train_vq_loss, _ = vqvae(images)
#             train_loss = F.mse_loss(reconstructed, images) + train_vq_loss
#             train_loss.backward()
#             optimizer.step()

#             total_train_loss += train_loss.item()
#             total_train_vq_loss += train_vq_loss.item()

#         avg_train_loss = total_train_loss / len(train_loader)
#         avg_train_vq_loss = total_train_vq_loss / len(train_loader)

#         vqvae.eval()
#         total_valid_loss, total_valid_vq_loss = 0, 0
#         real_features, fake_features = [], []

#         with torch.no_grad():
#             for val_images, _ in tqdm(test_loader, desc="Validating"):
#                 val_images = val_images.to(mnist_config["device"])
#                 recon_images, valid_vq_loss, _ = vqvae(val_images)
#                 valid_loss = F.mse_loss(recon_images, val_images) + valid_vq_loss
#                 total_valid_loss += valid_loss.item()
#                 total_valid_vq_loss += valid_vq_loss.item()

#                 real_features.append(get_features(val_images, resnet_model, mnist_config["device"]))
#                 fake_features.append(get_features(recon_images, resnet_model, mnist_config["device"]))

#         avg_valid_loss = total_valid_loss / len(test_loader)
#         avg_valid_vq_loss = total_valid_vq_loss / len(test_loader)
#         fid_score = calculate_fid(np.concatenate(real_features, axis=0), np.concatenate(fake_features, axis=0))

#         wandb.log({"epoch": epoch + 1, "train_loss": avg_train_loss, "train_vq_loss": avg_train_vq_loss, "valid_loss": avg_valid_loss, "valid_vq_loss": avg_valid_vq_loss, "fid_score": fid_score})
#         print(f"Epoch [{epoch+1}/{mnist_config['epochs']}], Train Loss: {avg_train_loss:.4f}, Valid Loss: {avg_valid_loss:.4f}, FID: {fid_score:.4f}")
        
#         if avg_valid_loss < best_valid_loss:
#             best_valid_loss = avg_valid_loss
#             torch.save(vqvae.state_dict(), mnist_config["checkpoint_path"])
#             print(f"✅ Checkpoint Saved: {mnist_config['checkpoint_path']} (Valid Loss: {best_valid_loss:.4f})")


# def test_mnist():
#     """체크포인트 로드 후 validation 과정에서 FID Score만 측정"""
#     # wandb.init(project="vqvae-mnist", config=mnist_config)

#     # 데이터 로더 불러오기 (검증 데이터셋만 사용)
#     _, test_loader = get_mnist_dataloader()

#     # 모델 및 ResNet Feature Extractor 초기화
#     vqvae = VQVAE_MNIST(
#         1, 16, mnist_config["latent_dim"], 
#         mnist_config["num_embeddings"], 
#         mnist_config["commitment_cost"]
#     ).to(mnist_config["device"])

#     resnet_model = ResNetFeatureExtractor().to(mnist_config["device"])
#     resnet_model.eval()

#     # 🔹 체크포인트 로드
#     checkpoint_path = mnist_config["checkpoint_path"]
#     if os.path.exists(checkpoint_path):
#         print(f"🔄 기존 체크포인트 로드: {checkpoint_path}")
#         vqvae.load_state_dict(torch.load(checkpoint_path, map_location=mnist_config["device"]))
#     else:
#         print(f"❌ 체크포인트 {checkpoint_path} 가 존재하지 않습니다.")
#         return
    
#     vqvae.eval()

#     # 🔹 Validation 과정에서 FID Score 측정
#     real_features, fake_features = [], []

#     with torch.no_grad():
#         for val_images, _ in tqdm(test_loader, desc="Calculating FID"):
#             val_images = val_images.to(mnist_config["device"])
#             recon_images, valid_vq_loss, _ = vqvae(val_images)

#             # ResNet을 통해 Feature 추출
#             real_features.append(get_features(val_images, resnet_model, mnist_config["device"]))
#             fake_features.append(get_features(recon_images, resnet_model, mnist_config["device"]))

#     # 🔹 FID Score 계산
#     fid_score = calculate_fid(np.concatenate(real_features, axis=0), np.concatenate(fake_features, axis=0))
#     print(f"✅ FID Score: {fid_score * 1e3:.4f}")
