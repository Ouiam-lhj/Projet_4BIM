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
    """Construit une fonction de décodage d'image à partir d'un chemin de fichier.

    Args:
        test (bool): Si True, retourne une fonction de décodage simple (image uniquement),
                     sinon retourne une fonction renvoyant un tuple (image, image) pour l'entraînement.
        out_size (tuple, optional): Taille de sortie de l'image redimensionnée. Par défaut (128, 128).

    Returns:
        Callable: Fonction de décodage adaptée au mode (test ou entraînement).
    """
    def decoder(path):
        """Lit une image à partir d'un chemin, la redimensionne et la normalise.

        Args:
            path (tf.Tensor): Chemin de l'image à charger.

        Returns:
            tf.Tensor: Image normalisée (float32, valeurs entre 0 et 1).
        """
        img = file_bytes = tf.io.read_file(path)
        img = tf.image.decode_jpeg(file_bytes, channels=3)  
        img = tf.image.resize(img, (128, 128))
        img = tf.cast(img, tf.float32) / 255.0
        return img
    def decoder_train(path):
        """Retourne une paire d'images identiques pour l'entraînement (entrée et cible identiques).

        Args:
            path (tf.Tensor): Chemin de l'image à charger.

        Returns:
            tuple: (image, image)
        """
        return decoder(path), decoder(path)

    return decoder if test else decoder_train

def build_dataset(paths, test=False, shuffle=1, batch_size=1):
    """Crée un objet `tf.data.Dataset` à partir de chemins d'images.

    Args:
        paths (array-like): Liste des chemins vers les images.
        test (bool): Si `True`, ne retourne que les images (pas de paires).
        shuffle (int): Taille du buffer de mélange.
        batch_size (int): Taille des batchs.

    Returns:
        tf.data.Dataset: Dataset prétraité prêt pour l'entraînement ou l'évaluation.
    """
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
    """Classe représentant un Autoencodeur Variationnel (VAE)."""
    def __init__(self):
        """Initialise les hyperparamètres et les modèles vides."""
        self.input_dim = (128,128,3)
        self.batch_size = 512
        self.z_dim = 200 # Dimension of the latent vector (z)
        self.learning_rate = 0.0005
        self.var_autoencoder_model = None
        self.var_encoder_model = None
        self.var_decoder_model = None

    def build(self):
        """Construit l'encodeur, le décodeur et le modèle VAE complet.

        Cette méthode crée les couches Keras, définit la fonction de perte VAE 
        (perte de reconstruction + divergence KL), puis compile le modèle.
        """
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
            """Échantillonnage de la variable latente z à partir de la distribution gaussienne.

            Args:
                args (tuple): Moyenne (z_mean) et log-variance (z_log_var) de la distribution latente.

            Returns:
                tf.Tensor: Vecteur latent échantillonné.
            """
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

        reconstruction_loss = Lambda(
        lambda inputs: binary_crossentropy(
                tf.reshape(inputs[0], [-1]), tf.reshape(inputs[1], [-1])
            ) * (128 * 128)
        )([input_encoder, output_vae])

        input_flat = tf.reshape(input_encoder, [-1])
        output_flat = tf.reshape(output_vae, [-1])

        # Calculate reconstruction loss
        reconstruction_loss = binary_crossentropy(input_flat, output_flat) * (128 * 128)
        reconstruction_loss = K.mean(reconstruction_loss)

        # Calculate KL divergence loss
        kl_loss = 1 + latent_log_var - K.square(latent_mu) - K.exp(latent_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5

        # Combine reconstruction loss and KL divergence loss
        vae_loss = K.mean(reconstruction_loss + kl_loss)

        # Add the loss to the model
        self.var_autoencoder_model.add_loss(vae_loss)
        self.var_autoencoder_model.add_metric(reconstruction_loss, name='reconstruction_loss')
        self.var_autoencoder_model.add_metric(kl_loss, name='kl_divergence_loss')

        # Compile the model
        optimizer = Adam(lr=self.learning_rate)
        self.var_autoencoder_model.compile(optimizer=optimizer)

        return None

    def get_varencoder(self):
        """Renvoie le modèle encodeur VAE.

        Returns:
            keras.Model or None: Modèle encodeur, ou `None` s'il n'a pas été construit.
        """
        if self.var_encoder_model is not None:
            return self.var_encoder_model
        else:
            print("Variable Encoder model has not been defined!")
            return None

    def get_vardecoder(self):
        """Renvoie le modèle décodeur VAE.

        Returns:
            keras.Model or None: Modèle décodeur, ou `None` s'il n'a pas été construit.
        """
        if self.var_decoder_model is not None:
            return self.var_decoder_model
        else:
            print("Variable Decoder model has not been defined!")
            return None
    
    def get_varautoencoder(self):
        """Renvoie le modèle complet VAE.

        Returns:
            keras.Model or None: Modèle autoencodeur complet, ou `None` s'il n'a pas été construit.
        """
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



def generate_images_from_data(data, var_autoencoder, output_folder="image_file"):
    """
    Génère 6 images à partir des données en utilisant un autoencodeur
    et les enregistre dans un dossier spécifié.

    Args:
        data (numpy.ndarray): Données d'entrée.
        var_autoencoder (Model): Modèle autoencodeur utilisé pour prédire les images.
        output_folder (str): Nom du dossier où enregistrer les images.

    Returns:
        list: Chemins des images générées.
    """
    os.makedirs(output_folder, exist_ok=True)
    image_paths = []

    for n in range(0, 12, 2):
        fig, ax = plt.subplots(figsize=(8, 8))
        image = var_autoencoder.predict(data[n])

        ax.imshow(np.squeeze(image))
        ax.axis("off")

        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

        image_path = os.path.join(output_folder, f'image_{n}.png')
        plt.savefig(image_path, bbox_inches='tight', pad_inches=0)
        image_paths.append(image_path)

        plt.close(fig)

    plt.show()
    return image_paths

generated_image_paths = generate_images_from_data(data, var_autoencoder)



def load_images_as_matrix(image_folder="image_file"):
    """
    Charge les images .png depuis un dossier et retourne une matrice RGB.

    Args:
        image_folder (str): Chemin du dossier contenant les images.

    Returns:
        np.ndarray: Matrice contenant les valeurs RGB des images.
    """
    image_files = sorted([
        os.path.join(image_folder, f) for f in os.listdir(image_folder)
        if f.endswith('.png')
    ])

    images = []
    for file in image_files:
        '''Renvoie une matrice contenant des valeurs RGB pour chaque pixel'''
        img = Image.open(file).convert('RGB')  
        img_array = np.array(img)
        images.append(img_array)

    image_matrix = np.stack(images)

    return image_matrix


#image_matrix = load_images_as_matrix("image_file")
# for i in range(len(image_matrix)):
#     plt.figure(figsize=(4, 4))
#     plt.imshow(image_matrix[i])
#     plt.axis('off')
#     plt.title(f"Image {i}")
#     plt.show()

def load_images_from_folder(images, image_size=(128, 128)):
    """Charge toutes les images du dossier image_file et les redimensionne."""
    
    for img in images:
        img = img.convert('RGB')
        img = img.resize(image_size)
        img_array = np.array(img) / 255.0  # Normalisation des valeurs RGB
        images.append(img_array)
    
    return np.array(images)


def mutate_latent_vector(latent_vector, mutation_strength=0.5):
    """Applique une mutation aléatoire au vecteur latent."""
    mutation = np.random.normal(0, mutation_strength, latent_vector.shape)
    mutated_latent_vector = latent_vector + mutation

    return mutated_latent_vector


def vae_generate_mutated_images(var_encoder, var_decoder, images, new_to_show=10, mutation_strength=0.5):
    selected_images = images[:new_to_show]
    image_arrays = np.stack([np.array(img) for img in selected_images])
    latent_vectors = var_encoder.predict(image_arrays)
    mutated_latent_vectors = np.array([mutate_latent_vector(latent_vector, mutation_strength) for latent_vector in latent_vectors])
    mutated_images = var_decoder.predict(mutated_latent_vectors)
    image_list = [Image.fromarray((np.squeeze(img)).astype(np.uint8)) for img in mutated_images]
    
    return image_list
    




# def vae_generate_images(new_to_show=10):
#     random_codes = np.random.normal(size=(new_to_show, 200))
#     new_faces = var_decoder.predict(np.array(random_codes))

#     fig = plt.figure(figsize=(30, 15))

#     for i in range(new_to_show):
#         ax = fig.add_subplot(6, 10, i+1)
#         ax.imshow(new_faces[i])
#         ax.axis('off')
#     plt.show()


# vae_generate_images(10)