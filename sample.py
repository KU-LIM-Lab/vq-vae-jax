import os
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import numpy as np
from PIL import Image
from modules import VQVAE, GatedPixelCNN

# 경로 설정
vqvae_checkpoint = "/root/limlab/yeongyu/pytorch-vqvae/models/models/vqvae/best.pkl"
pixelcnn_prior_checkpoint = "/root/limlab/yeongyu/pytorch-vqvae/models/models/pixelcnn_prior/prior.pkl"
real_image_folder = '/root/limlab/yeongyu/pytorch-vqvae/generated_samples/real'
fake_image_folder = '/root/limlab/yeongyu/pytorch-vqvae/generated_samples/fake'
os.makedirs(real_image_folder, exist_ok=True)
os.makedirs(fake_image_folder, exist_ok=True)

# 하이퍼파라미터 설정
k = 512
hidden_size_vae = 256
hidden_size_prior = 64
num_layers = 15
num_classes = 100  # 전체 클래스 수
image_shape = (32, 32)
batch_size = 32
num_images_per_class = 96  # 클래스 0에 대해 생성할 이미지 수

# JAX 디바이스 설정
rng = jax.random.PRNGKey(0)

# VQ-VAE 모델 로드
vqvae = VQVAE(
    in_channels=3, hidden_channels=hidden_size_vae, latent_dim=k,
    num_residual_layers=4, residual_hidden_channels=32,
    num_embeddings=k, commitment_cost=1.0
)
vqvae_params = vqvae.init(rng, jnp.ones((1, 3, 128, 128)))['params']

# PixelCNN Prior 모델 로드
pixelcnn_prior = GatedPixelCNN(input_dim=k, dim=hidden_size_prior, n_layers=num_layers, n_classes=num_classes)
pixelcnn_params = pixelcnn_prior.init(rng, jnp.ones((1, *image_shape), dtype=jnp.int32), jnp.ones((1,)))['params']

# 클래스별 폴더 경로 생성
cls = 0
print(f"클래스 {cls}에 대한 이미지 생성 중...")
cls_fake_image_folder = os.path.join(fake_image_folder, f'class_{cls}')
os.makedirs(cls_fake_image_folder, exist_ok=True)

@jax.jit
def generate_samples(label, pixelcnn_params):
    latents = pixelcnn_prior.apply({'params': pixelcnn_params}, label, shape=image_shape, batch_size=batch_size)
    return latents

@jax.jit
def decode_samples(latents, vqvae_params):
    generated_images = vqvae.apply({'params': vqvae_params}, latents)
    return generated_images

# 이미지 생성 및 저장
for batch_idx in range(num_images_per_class // batch_size):
    label = np.full((batch_size,), cls, dtype=np.int32)
    latents = generate_samples(label, pixelcnn_params)
    latents = np.array(latents)

    # VQ-VAE 디코더로 복원
    generated_images = decode_samples(latents, vqvae_params)
    generated_images = np.array(generated_images)

    # 생성된 이미지 저장
    for idx in range(generated_images.shape[0]):
        image_idx = batch_idx * batch_size + idx
        image = (generated_images[idx].transpose(1, 2, 0) * 255).astype(np.uint8)
        image = Image.fromarray(image)
        image.save(f"{cls_fake_image_folder}/image_{image_idx}.png")

print(f"클래스 {cls}에 대해 {num_images_per_class}개 이미지 생성 완료.")

# 원본 이미지 저장 (테스트용 또는 FID 계산용)
from torchvision import datasets, transforms
import torch

transform = transforms.Compose([
    transforms.RandomResizedCrop(128),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
test_dataset = datasets.ImageFolder("/root/limlab/yeongyu/pytorch-vqvae/datasets/miniimagenet/test", transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)

cls_real_image_folder = os.path.join(real_image_folder, f'class_{cls}')
os.makedirs(cls_real_image_folder, exist_ok=True)

real_image_count = 0

for idx, (real_image, label) in enumerate(test_loader):
    if label.item() != cls:
        continue

    real_image = real_image.numpy().transpose(1, 2, 0)  # [C, H, W] -> [H, W, C]
    real_image = (real_image * 255).astype(np.uint8)
    image = Image.fromarray(real_image)
    image.save(f"{cls_real_image_folder}/image_{real_image_count}.png")
    print(f"클래스 {cls}의 원본 이미지 저장: {cls_real_image_folder}/image_{real_image_count}.png")

    real_image_count += 1
    if real_image_count >= num_images_per_class:
        break

print(f"클래스 {cls}에 대한 원본 이미지 {real_image_count}장 저장 완료.")
print(f"FID 계산 명령어:")
print(f"pytorch-fid {real_image_folder} {fake_image_folder}")