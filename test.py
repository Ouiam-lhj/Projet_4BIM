import numpy as np
import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, Reshape, Lambda, Activation, LeakyReLU
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
from glob import glob
from PIL import Image
import tensorflow as tf


print(" Liste des GPUs disponibles :", tf.config.list_physical_devices('GPU'))

try:
    tf.debugging.set_log_device_placement(True)
    with tf.device('/device:GPU:0'):
        print(" Exécution d'un test sur GPU")
        a = tf.constant([[1.0, 2.0, 3.0]])
        b = tf.constant([[4.0], [5.0], [6.0]])
        c = tf.matmul(a, b)
        print("Résultat du test :", c.numpy())
except Exception as e:
    print(" Erreur d'utilisation du GPU :", e)


WEIGHTS_FOLDER = './weights/'
DATA_FOLDER = './img_align_celeba/'
Z_DIM = 200

if not os.path.exists(WEIGHTS_FOLDER):
  os.makedirs(os.path.join(WEIGHTS_FOLDER,"AE"))
  os.makedirs(os.path.join(WEIGHTS_FOLDER,"VAE"))

filenames = np.array(glob(os.path.join(DATA_FOLDER, '*.jpg')))
NUM_IMAGES = len(filenames)
print("Total number of images : " + str(NUM_IMAGES))


def build_decoder(test=False, out_size=(128, 128)):
    def decoder(path):
        img = file_bytes = tf.io.read_file(path)
        img = tf.image.decode_jpeg(file_bytes, channels=3)  
        img = tf.image.resize(img, (128, 128))
        img = tf.cast(img, tf.float32) / 255.0
        return img
    def decoder_train(path):
        return decoder(path), decoder(path)

    return decoder if test else decoder_train

def build_dataset(paths, test=False, shuffle=1, batch_size=1):
    AUTO = tf.data.experimental.AUTOTUNE
    decoder = build_decoder(test)

    dset = tf.data.Dataset.from_tensor_slices(paths)
    dset = dset.map(decoder, num_parallel_calls=AUTO)
    
    dset = dset.shuffle(shuffle)
    dset = dset.batch(batch_size)
    return dset


train_paths, valid_paths, _, _ = train_test_split(filenames, filenames, test_size=0.2, shuffle=True)

train_dataset = build_dataset(train_paths, batch_size=128)
valid_dataset = build_dataset(valid_paths, batch_size=128)


class VariableAutoencoder:
    def __init__(self):
        self.input_dim = (128,128,3)
        self.batch_size = 512
        self.z_dim = 200 # Dimension of the latent vector (z)
        self.learning_rate = 0.0005
        self.var_autoencoder_model = None
        self.var_encoder_model = None
        self.var_decoder_model = None

    def build(self):
        #Encoder
        input_encoder = Input(shape=(self.input_dim))
        x = Conv2D(32, kernel_size=(3, 3), strides = 2, padding='same', name='encoder_conv2d_1')(input_encoder)
        x = LeakyReLU()(x)
        x = Conv2D(64, kernel_size=(3, 3), strides = 2, padding='same', name='encoder_conv2d_2')(x)
        x = LeakyReLU()(x)
        x = Conv2D(64, kernel_size=(3, 3), strides = 2, padding='same', name='encoder_conv2d_3')(x)
        x = LeakyReLU()(x)
        x = Conv2D(64, kernel_size=(3, 3), strides = 2, padding='same', name='encoder_conv2d_4')(x)
        volumeSize = K.int_shape(x)
        x = Flatten()(x)

        latent_mu = Dense(self.z_dim, name='latent_mean')(x)
        latent_log_var = Dense(self.z_dim, name='latent_log_var')(x)
        
        def sampling(args=None):
            z_mean, z_log_var = args
            batch = K.shape(z_mean)[0]

            epsilon = K.random_normal(shape=(batch, self.z_dim))
            return z_mean + K.exp(0.5 * z_log_var) * epsilon
        
        latent_sample = Lambda(sampling)([latent_mu, latent_log_var])
        self.var_encoder_model = Model(input_encoder, latent_sample, name='encoder')

        latent_input = Input(shape=(self.z_dim,), name='decoder_input')
        x = Dense(np.prod(volumeSize[1:]))(latent_input)
        x = Reshape((volumeSize[1], volumeSize[2], volumeSize[3]))(x)
        x = Conv2DTranspose(64, kernel_size=(3, 3), strides=2, padding='same', name='conv2d_1')(x)
        x = LeakyReLU()(x)
        x = Conv2DTranspose(64, kernel_size=(3, 3), strides=2, padding='same', name='conv2d_2')(x)
        x = LeakyReLU()(x)
        x = Conv2DTranspose(32, kernel_size=(3, 3), strides=2, padding='same', name='conv2d_3')(x)
        x = LeakyReLU()(x)
        x = Conv2DTranspose(3, kernel_size=(3, 3), strides=2, padding='same', name='conv2d_4')(x)
        output_decoder = Activation('sigmoid')(x)

        self.var_decoder_model = Model(latent_input, output_decoder, name='decoder')

        output_vae = self.var_decoder_model(self.var_encoder_model(input_encoder))
        self.var_autoencoder_model = Model(input_encoder, output_vae, name ='variable_autoencoder')

        reconstruction_loss = binary_crossentropy(input_encoder, output_vae) * (128 * 128)
        reconstruction_loss = K.mean(reconstruction_loss)

        kl_loss = 1 + latent_log_var - K.square(latent_mu) - K.exp(latent_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5

        vae_loss = K.mean(reconstruction_loss + kl_loss)

        self.var_autoencoder_model.add_loss(vae_loss)  
        self.var_autoencoder_model.add_metric(reconstruction_loss, name='reconstruction_loss')
        self.var_autoencoder_model.add_metric(kl_loss, name='kl_divergence_loss')

        optimizer = Adam(lr = self.learning_rate)

        self.var_autoencoder_model.compile(optimizer=optimizer)

        return None

    def get_varencoder(self):
        if self.var_encoder_model is not None:
            return self.var_encoder_model
        else:
            print("Variable Encoder model has not been defined!")
            return None

    def get_vardecoder(self):
        if self.var_decoder_model is not None:
            return self.var_decoder_model
        else:
            print("Variable Decoder model has not been defined!")
            return None
    
    def get_varautoencoder(self):
        if self.var_autoencoder_model is not None:
            return self.var_autoencoder_model
        else:
            print("Variable Autoencoder model has not been defined!")
            return None


var_model = VariableAutoencoder()
var_model.build()

var_autoencoder = var_model.get_varautoencoder()
var_encoder = var_model.get_varencoder()
var_decoder = var_model.get_vardecoder()
var_autoencoder.summary()

test_dataset = build_dataset(valid_paths, test=True)
var_autoencoder.load_weights(os.path.join(WEIGHTS_FOLDER, 'VAE/vae_last_model.h5'))


data = list(test_dataset.take(20))


output_folder = "image_file"
os.makedirs(output_folder, exist_ok=True)


for n in range(0, 14, 2):
    fig, ax = plt.subplots(figsize=(8, 8))
    image = var_autoencoder.predict(data[n])

    ax.imshow(np.squeeze(image))
    ax.axis("off")

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Chemin complet vers l'image dans le dossier
    image_path = os.path.join(output_folder, f'image_{n}.png')
    plt.savefig(image_path, bbox_inches='tight', pad_inches=0)

    plt.close(fig)

plt.show()

from PIL import Image

image_folder = "image_file"

image_files = sorted([
    os.path.join(image_folder, f) for f in os.listdir(image_folder)
    if f.endswith('.png')
])

images = []
for file in image_files:
    img = Image.open(file).convert('RGB')  
    img_array = np.array(img)
    images.append(img_array)

image_matrix = np.stack(images)

print("Shape of image matrix:", image_matrix.shape)


