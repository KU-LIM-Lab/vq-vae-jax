from train import train_imagenet, train_mnist
import argparse
import jax

def main():
    parser = argparse.ArgumentParser(description="Train VQ-VAE on ImageNet or MNIST using JAX & Flax")
    parser.add_argument("--dataset", type=str, choices=["imagenet", "mnist"], required=True, help="Choose dataset: imagenet or mnist")
    args = parser.parse_args()
    
    print(f"VQ-VAE Training 시작! Dataset: {args.dataset} (Using JAX & Flax)")
    print("Available devices:", jax.devices())
    
    if args.dataset == "imagenet":
        train_imagenet()
    else:
        train_mnist()

if __name__ == "__main__":
    main()
