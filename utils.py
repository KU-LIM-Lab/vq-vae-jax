import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from scipy.linalg import sqrtm
from torchvision.models import resnet18
from flax.training.common_utils import shard

# 🔹 ResNet-18 Feature Extractor (MNIST를 위한 1채널 지원)
class ResNetFeatureExtractor(nn.Module):
    def setup(self):
        self.resnet = resnet18(pretrained=True)
        self.resnet.conv1 = nn.Conv(1, 64, kernel_size=(7, 7), strides=(2, 2), padding=((3, 3), (3, 3)), use_bias=False)
        self.resnet.fc = lambda x: x  # Fully Connected Layer 제거 (Feature 추출)

    def __call__(self, x):
        return self.resnet(x)

# 🔹 Feature 추출 함수
def get_features(images, model, params):
    images = jnp.array(images)
    features = model.apply(params, images)
    return np.array(features)

def calculate_fid(real_features, fake_features):
    """Frechet Inception Distance (FID) 계산"""
    mu_real, sigma_real = jnp.mean(real_features, axis=0), jnp.cov(real_features, rowvar=False)
    mu_fake, sigma_fake = jnp.mean(fake_features, axis=0), jnp.cov(fake_features, rowvar=False)

    diff = mu_real - mu_fake
    cov_mean, _ = sqrtm(sigma_real @ sigma_fake, disp=False)

    if jnp.iscomplexobj(cov_mean):
        cov_mean = cov_mean.real  # 허수 제거

    fid_score = jnp.sum(diff**2) + jnp.trace(sigma_real + sigma_fake - 2 * cov_mean)
    return fid_score
