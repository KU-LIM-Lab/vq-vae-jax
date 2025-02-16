import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision.models import inception_v3
import os
import wandb
from tqdm import tqdm

from model_imagenet import VQVAE as VQVAE_Imagenet
from model_mnist import VQVAE as VQVAE_MNIST
from dataset import get_imagenet_dataloader, get_mnist_dataloader
from config import imagenet_config, mnist_config
from utils import ResNetFeatureExtractor, get_features, calculate_fid

def train_imagenet():
    wandb.init(project="vqvae-imagenet", config=imagenet_config)
    train_loader, test_loader = get_imagenet_dataloader()
    vqvae = VQVAE_Imagenet(3, 128, imagenet_config["latent_dim"], 2, 32, imagenet_config["num_embeddings"], imagenet_config["commitment_cost"]).to(imagenet_config["device"])

    inception_model = inception_v3(pretrained=True, transform_input=False).to(imagenet_config["device"])
    inception_model.fc = torch.nn.Identity()
    inception_model.eval()

    optimizer = optim.Adam(vqvae.parameters(), lr=imagenet_config["lr"])
    best_valid_loss = float("inf")

    for epoch in range(imagenet_config["epochs"]):
        vqvae.train()
        total_train_loss, total_train_vq_loss = 0, 0

        for images, _ in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            images = images.to(imagenet_config["device"])
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
        real_features, fake_features = [], []

        with torch.no_grad():
            for val_images, _ in tqdm(test_loader, desc="Validating"):
                val_images = val_images.to(imagenet_config["device"])
                recon_images, valid_vq_loss = vqvae(val_images)
                valid_loss = F.mse_loss(recon_images, val_images) + valid_vq_loss

                total_valid_loss += valid_loss.item()
                total_valid_vq_loss += valid_vq_loss.item()

                real_features.append(get_features(val_images, inception_model, imagenet_config["device"]))
                fake_features.append(get_features(recon_images, inception_model, imagenet_config["device"]))

        avg_valid_loss = total_valid_loss / len(test_loader)
        avg_valid_vq_loss = total_valid_vq_loss / len(test_loader)
        fid_score = calculate_fid(np.concatenate(real_features, axis=0), np.concatenate(fake_features, axis=0))

        wandb.log({"epoch": epoch + 1, "train_loss": avg_train_loss, "train_vq_loss": avg_train_vq_loss, "valid_loss": avg_valid_loss, "valid_vq_loss": avg_valid_vq_loss, "fid_score": fid_score})

        print(f"Epoch [{epoch+1}/{imagenet_config['epochs']}], Train Loss: {avg_train_loss:.4f}, Valid Loss: {avg_valid_loss:.4f}, FID: {fid_score:.2f}")
        
        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            torch.save(vqvae.state_dict(), imagenet_config["checkpoint_path"])
            print(f"✅ Checkpoint Saved: {imagenet_config['checkpoint_path']} (Valid Loss: {best_valid_loss:.4f})")

def train_mnist():
    wandb.init(project="vqvae-mnist", config=mnist_config)
    train_loader, test_loader = get_mnist_dataloader()
    vqvae = VQVAE_MNIST(1, 16, mnist_config["latent_dim"], mnist_config["num_embeddings"], mnist_config["commitment_cost"]).to(mnist_config["device"])
    resnet_model = ResNetFeatureExtractor().to(mnist_config["device"])
    resnet_model.eval()

    optimizer = optim.Adam(vqvae.parameters(), lr=mnist_config["lr"])
    best_valid_loss = float("inf")

    for epoch in range(mnist_config["epochs"]):
        vqvae.train()
        total_train_loss, total_train_vq_loss = 0, 0

        for images, _ in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            images = images.to(mnist_config["device"])
            optimizer.zero_grad()
            reconstructed, train_vq_loss, _ = vqvae(images)
            train_loss = F.mse_loss(reconstructed, images) + train_vq_loss
            train_loss.backward()
            optimizer.step()

            total_train_loss += train_loss.item()
            total_train_vq_loss += train_vq_loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_train_vq_loss = total_train_vq_loss / len(train_loader)

        vqvae.eval()
        total_valid_loss, total_valid_vq_loss = 0, 0
        real_features, fake_features = [], []

        with torch.no_grad():
            for val_images, _ in tqdm(test_loader, desc="Validating"):
                val_images = val_images.to(mnist_config["device"])
                recon_images, valid_vq_loss, _ = vqvae(val_images)
                valid_loss = F.mse_loss(recon_images, val_images) + valid_vq_loss
                total_valid_loss += valid_loss.item()
                total_valid_vq_loss += valid_vq_loss.item()

                real_features.append(get_features(val_images, resnet_model, mnist_config["device"]))
                fake_features.append(get_features(recon_images, resnet_model, mnist_config["device"]))

        avg_valid_loss = total_valid_loss / len(test_loader)
        avg_valid_vq_loss = total_valid_vq_loss / len(test_loader)
        fid_score = calculate_fid(np.concatenate(real_features, axis=0), np.concatenate(fake_features, axis=0))

        wandb.log({"epoch": epoch + 1, "train_loss": avg_train_loss, "train_vq_loss": avg_train_vq_loss, "valid_loss": avg_valid_loss, "valid_vq_loss": avg_valid_vq_loss, "fid_score": fid_score})
        print(f"Epoch [{epoch+1}/{mnist_config['epochs']}], Train Loss: {avg_train_loss:.4f}, Valid Loss: {avg_valid_loss:.4f}, FID: {fid_score:.4f}")
        
        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            torch.save(vqvae.state_dict(), mnist_config["checkpoint_path"])
            print(f"✅ Checkpoint Saved: {mnist_config['checkpoint_path']} (Valid Loss: {best_valid_loss:.4f})")
