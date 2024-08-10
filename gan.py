import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Make sure to set up TensorBoard logs
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


# Set parameters
img_height, img_width = 64, 64  # Adjust size as needed
batch_size = 32

# Data augmentation and normalization
datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)

# Replace with the path to your dataset
dataset_path = 'path_to_crawford_cats_dataset'
train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode=None,
    shuffle=True
)

def build_generator():
    model = Sequential()
    model.add(layers.Dense(128 * 16 * 16, activation="relu", input_dim=100))
    model.add(layers.Reshape((16, 16, 128)))
    model.add(layers.Conv2DTranspose(128, kernel_size=5, strides=2, padding="same", activation="relu"))
    model.add(layers.Conv2DTranspose(64, kernel_size=5, strides=2, padding="same", activation="relu"))
    model.add(layers.Conv2DTranspose(3, kernel_size=5, strides=1, padding="same", activation="tanh"))
    return model

def build_discriminator():
    model = Sequential()
    model.add(layers.Conv2D(64, kernel_size=5, strides=2, padding="same", input_shape=(64, 64, 3)))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.3))
    model.add(layers.Conv2D(128, kernel_size=5, strides=2, padding="same"))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.3))
    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation="sigmoid"))
    return model

def build_gan(generator, discriminator):
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    return model

generator = build_generator()
discriminator = build_discriminator()
discriminator.compile(optimizer=Adam(0.0002, 0.5), loss='binary_crossentropy', metrics=['accuracy'])

gan = build_gan(generator, discriminator)
gan.compile(optimizer=Adam(0.0002, 0.5), loss='binary_crossentropy')


def train_gan(epochs, batch_size=32):
    half_batch = batch_size // 2
    
    for epoch in range(epochs):
        # Train Discriminator
        imgs, _ = next(train_generator)
        noise = np.random.normal(0, 1, (half_batch, 100))
        generated_imgs = generator.predict(noise)
        
        d_loss_real = discriminator.train_on_batch(imgs, np.ones((half_batch, 1)))
        d_loss_fake = discriminator.train_on_batch(generated_imgs, np.zeros((half_batch, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Train Generator
        noise = np.random.normal(0, 1, (batch_size, 100))
        g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))
        
        # Print the progress
        print(f"{epoch}/{epochs} [D loss: {d_loss[0]} | D Accuracy: {100*d_loss[1]}] [G loss: {g_loss}]")
        
        # Save generated images periodically
        if epoch % 100 == 0:
            save_generated_images(epoch)

def save_generated_images(epoch, examples=10, dim=(1, 10), figsize=(10, 1)):
    noise = np.random.normal(0, 1, (examples, 100))
    generated_images = generator.predict(noise)
    generated_images = 0.5 * generated_images + 0.5  # Rescale images to [0, 1]

    fig, ax = plt.subplots(dim[0], dim[1], figsize=figsize)
    count = 0
    for i in range(dim[0]):
        for j in range(dim[1]):
            ax[i, j].imshow(generated_images[count])
            ax[i, j].axis('off')
            count += 1
    plt.savefig(f"gan_generated_image_{epoch}.png")
    plt.close()

train_gan(epochs=10000, batch_size=batch_size)

# tensorboard --logdir=logs/fit

