import os
import kagglehub
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from config import imagenet_config, cifar10_config, mnist_config


def get_cifar10_dataloader():
    transform = transforms.Compose([
        transforms.Resize((cifar10_config["image_size"], cifar10_config["image_size"])),  # CIFAR-10은 32x32로 유지
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # VQ-VAE 논문에서 사용한 정규화
    ])

    train_dataset = datasets.CIFAR10(root="./cifar10_data", train=True, transform=transform, download=True)
    test_dataset = datasets.CIFAR10(root="./cifar10_data", train=False, transform=transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=cifar10_config["batch_size"], shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=cifar10_config["batch_size"], shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, test_loader

def get_imagenet_dataloader():
    path = kagglehub.dataset_download("ifigotin/imagenetmini-1000")
    DATASET_PATH = os.path.join(path, "imagenet-mini")
    TRAIN_PATH = os.path.join(DATASET_PATH, "train")
    VAL_PATH = os.path.join(DATASET_PATH, "val")

    transform = transforms.Compose([
        transforms.Resize((imagenet_config["image_size"], imagenet_config["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = ImageFolder(root=TRAIN_PATH, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=imagenet_config["batch_size"], shuffle=True, num_workers=4)

    test_dataset = ImageFolder(root=VAL_PATH, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=imagenet_config["batch_size"], shuffle=False, num_workers=4)

    return train_loader, test_loader

def get_mnist_dataloader():
    transform = transforms.Compose([
        transforms.Resize(mnist_config["image_size"]),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    train_set = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    test_set = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

    train_loader = DataLoader(train_set, batch_size=mnist_config["batch_size"], shuffle=True, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=mnist_config["batch_size"], shuffle=False, num_workers=4)

    return train_loader, test_loader
