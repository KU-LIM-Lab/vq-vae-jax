import argparse
import os

from train import train_imagenet, train_cifar10, fid_cifar10, train_mnist

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train, Evaluate, or Reconstruct VQ-VAE on ImageNet, CIFAR-10, or MNIST")

    # 기본 데이터셋을 CIFAR-10로 설정
    parser.add_argument("--dataset", type=str, choices=["imagenet", "cifar10", "mnist"], default="cifar10", help="Choose dataset: imagenet, cifar10, or mnist")
    parser.add_argument("--mode", type=str, choices=["train", "fid", "reconstruct"], default="train", help="Choose mode: train, fid, or reconstruct")

    args = parser.parse_args()

    print(f"VQ-VAE 실행! Dataset: {args.dataset}, Mode: {args.mode}")

    if args.mode == "train":
        if args.dataset == "imagenet":
            train_imagenet()
        elif args.dataset == "cifar10":
            train_cifar10()
        else:
            train_mnist()
    
    elif args.mode == "fid":
        if args.dataset == "imagenet":
            print("🔹 ImageNet용 FID 평가 코드는 아직 구현되지 않았습니다.")
        elif args.dataset == "cifar10":
            fid_cifar10()
        else:
            print("🔹 MNIST용 FID 평가 코드는 아직 구현되지 않았습니다.")

    elif args.mode == "reconstruct":
        os.system(f"python reconstruction.py --dataset {args.dataset}")