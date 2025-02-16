import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from scipy.linalg import sqrtm


# 🔹 ResNet-18 Feature Extractor (MNIST를 위한 1채널 지원)
class ResNetFeatureExtractor(nn.Module):
    def __init__(self):
        super(ResNetFeatureExtractor, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 1채널 지원
        self.resnet.fc = nn.Identity()  # Fully Connected Layer 제거 (Feature 추출)

    def forward(self, x):
        return self.resnet(x)

# 🔹 Feature 추출 함수
def get_features(images, model, device):
    images = images.to(device)
    with torch.no_grad():
        features = model(images)
    return features.cpu().numpy()

def calculate_fid(real_features, fake_features):
    """Frechet Inception Distance (FID) 계산"""
    mu_real, sigma_real = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)
    mu_fake, sigma_fake = np.mean(fake_features, axis=0), np.cov(fake_features, rowvar=False)

    diff = mu_real - mu_fake
    cov_mean, _ = sqrtm(sigma_real @ sigma_fake, disp=False)

    if np.iscomplexobj(cov_mean):
        cov_mean = cov_mean.real  # 허수 제거

    fid_score = np.sum(diff**2) + np.trace(sigma_real + sigma_fake - 2 * cov_mean)
    return fid_score
