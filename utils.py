import jax
import jax.numpy as jnp
import tensorflow as tf
import tensorflow_datasets as tfds


def get_dataset(args):
    train_ds: tf.data.Dataset = tfds.load('cifar10', split='train')
    test_ds: tf.data.Dataset = tfds.load('cifar10', split='test')

    # normalize datasetset
    train_ds = train_ds.map(
        lambda sample: {
            'image': tf.cast(sample['image'], tf.float32) / 255,
            'label': sample['label'],
        }
    )  
    test_ds = test_ds.map(
        lambda sample: {
            'image': tf.cast(sample['image'], tf.float32) / 255,
            'label': sample['label'],
        }
    )
    
    # Create a shuffled dataset by allocating a buffer size of 1024 to randomly draw elements from.
    # train_ds = train_ds.repeat().shuffle(len(train_ds), seed=0)
    train_ds = train_ds.shuffle(len(train_ds), seed=0)
    # Group into batches of `batch_size` and skip incomplete batches, prefetch the next sample to improve latency.
    # train_ds = train_ds.batch(args.batch_size, drop_remainder=True).take(args.train_steps).prefetch(1)
    # train_ds = train_ds.batch(args.batch_size, drop_remainder=False).prefetch(1)
    # Group into batches of `batch_size`, prefetch the next sample to improve latency.
    test_ds = test_ds.batch(args.batch_size, drop_remainder=False).prefetch(1)
    return train_ds, test_ds
    

# def calculate_fid(real_features, fake_features):
#     """Frechet Inception Distance (FID) 계산"""
#     mu_real, sigma_real = jnp.mean(real_features, axis=0), jnp.cov(real_features, rowvar=False)
#     mu_fake, sigma_fake = jnp.mean(fake_features, axis=0), jnp.cov(fake_features, rowvar=False)

#     diff = mu_real - mu_fake
#     cov_mean, _ = sqrtm(sigma_real @ sigma_fake, disp=False)

#     if jnp.iscomplexobj(cov_mean):
#         cov_mean = cov_mean.real  # 허수 제거

#     fid_score = jnp.sum(diff**2) + jnp.trace(sigma_real + sigma_fake - 2 * cov_mean)
#     return fid_score
