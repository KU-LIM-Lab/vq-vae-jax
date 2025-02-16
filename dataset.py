import os
import kagglehub
import jax.numpy as jnp
import tensorflow as tf
import tensorflow_datasets as tfds
from config import imagenet_config, mnist_config

def get_imagenet_dataloader():
    path = kagglehub.dataset_download("ifigotin/imagenetmini-1000")
    DATASET_PATH = os.path.join(path, "imagenet-mini")
    TRAIN_PATH = os.path.join(DATASET_PATH, "train")
    VAL_PATH = os.path.join(DATASET_PATH, "val")

    def preprocess_image(image, label):
        image = tf.image.resize(image, (imagenet_config["image_size"], imagenet_config["image_size"]))
        image = tf.cast(image, tf.float32) / 255.0
        image = (image - jnp.array([0.485, 0.456, 0.406])) / jnp.array([0.229, 0.224, 0.225])
        return image, label

    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        TRAIN_PATH, image_size=(imagenet_config["image_size"], imagenet_config["image_size"]), batch_size=imagenet_config["batch_size"], shuffle=True
    ).map(preprocess_image)

    test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        VAL_PATH, image_size=(imagenet_config["image_size"], imagenet_config["image_size"]), batch_size=imagenet_config["batch_size"], shuffle=False
    ).map(preprocess_image)

    return train_dataset, test_dataset

def get_mnist_dataloader():
    def preprocess_mnist(image, label):
        image = tf.image.resize(image, (mnist_config["image_size"], mnist_config["image_size"]))
        image = tf.cast(image, tf.float32) / 255.0
        image = (image - 0.5) / 0.5
        return image, label

    train_dataset, test_dataset = tfds.load("mnist", split=["train", "test"], as_supervised=True)
    train_dataset = train_dataset.map(preprocess_mnist).batch(mnist_config["batch_size"]).shuffle(10000)
    test_dataset = test_dataset.map(preprocess_mnist).batch(mnist_config["batch_size"]) 

    return train_dataset, test_dataset
