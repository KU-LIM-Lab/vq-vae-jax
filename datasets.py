import os
import tensorflow as tf
import tensorflow_datasets as tfds
from sklearn.model_selection import train_test_split

class MiniImagenet:
    def __init__(self, root, train=True, test=False, valid=False, transform=None, valid_ratio=0.2):
        self.root = os.path.abspath(root)
        self.transform = transform

        if train:
            self.data_path = os.path.join(self.root, "train")
        elif test:
            self.data_path = os.path.join(self.root, "test")
        else:
            raise ValueError("Only train or test can be selected.")

        # Load images and labels using tensorflow_datasets
        full_dataset = tfds.load('imagenet_v2', split='train', data_dir=self.data_path)
        full_dataset = full_dataset.map(self._preprocess)

        # Split the dataset into train and validation sets
        if train and valid_ratio > 0:
            total_size = len(full_dataset)
            train_size = int((1 - valid_ratio) * total_size)
            valid_size = total_size - train_size
            train_dataset = full_dataset.take(train_size)
            valid_dataset = full_dataset.skip(train_size)
            self.train_dataset = train_dataset.batch(32).prefetch(tf.data.AUTOTUNE)
            self.valid_dataset = valid_dataset.batch(32).prefetch(tf.data.AUTOTUNE)
        else:
            self.train_dataset = full_dataset.batch(32).prefetch(tf.data.AUTOTUNE)
            self.valid_dataset = None

    def _preprocess(self, sample):
        image = sample['image']
        label = sample['label']
        if self.transform:
            image = self.transform(image)
        else:
            image = tf.image.resize(image, [128, 128])
            image = tf.cast(image, tf.float32) / 255.0
        return image, label

    def __getitem__(self, index):
        if self.valid_dataset is None:
            return next(iter(self.train_dataset.skip(index).take(1)))
        return next(iter(self.valid_dataset.skip(index).take(1)))

    def __len__(self):
        return len(self.train_dataset) if self.valid_dataset is None else len(self.valid_dataset)
