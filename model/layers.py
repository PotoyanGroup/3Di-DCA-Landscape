# ref: https://github.com/morcoslab/LGL-VAE.git
import tensorflow as tf
from tensorflow import keras


class Sampling(keras.layers.Layer):
    """Subsample and locate layer by applying gaussian blur"""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]  # Dynamically determine batch size
        dim = tf.shape(z_mean)[1]  # Dynamically determine latent space dimension

        # Generate random noise
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon  # Reparameterization trick