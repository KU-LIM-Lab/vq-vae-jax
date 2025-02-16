from train import train_imagenet, train_mnist
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VQ-VAE on ImageNet or MNIST")
    parser.add_argument("--dataset", type=str, choices=["imagenet", "mnist"], required=True, help="Choose dataset: imagenet or mnist")
    args = parser.parse_args()
    
    print(f"VQ-VAE Training 시작! Dataset: {args.dataset}")
    if args.dataset == "imagenet":
        train_imagenet()
    else:
        train_mnist()
