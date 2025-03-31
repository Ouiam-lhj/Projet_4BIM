import numpy as np

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

import tensorflow as tf

print("🔍 Liste des GPUs disponibles :", tf.config.list_physical_devices('GPU'))

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


train_paths, valid_paths, _, _ = train_test_split(filenames, filenames, test_size=0.5, shuffle=True)

train_dataset = build_dataset(train_paths, batch_size=128)
valid_dataset = build_dataset(valid_paths, batch_size=128)


class ConvAutoencoder:
    def __init__(self):
        self.input_dim = (128,128,3)
        self.batch_size = 512
        self.latentDim = 200
        self.z_dim = 200 # Dimension of the latent vector (z)
        self.autoencoder_model = None
        self.encoder_model = None
        self.decoder_model = None

    #@staticmethod
    def build(self):
        inputs = Input(shape = self.input_dim)
        x = inputs
        
        filters=(32, 64, 64, 64)
        
        for index, f in enumerate(filters):
            x = Conv2D(f, (3,3), strides=2, padding="same", name="conv2dtranspose_" + str(index))(x)
            x = LeakyReLU()(x)
        
        volumeSize = K.int_shape(x)
        x = Flatten()(x)
        latent = Dense(self.latentDim)(x)
        self.encoder_model = Model(inputs, latent, name = "encoder")
        
        latentInputs = Input(shape=(self.latentDim,))
        x = Dense(np.prod(volumeSize[1:]))(latentInputs)
        x = Reshape((volumeSize[1], volumeSize[2], volumeSize[3]))(x)
        
        for f in [64, 64, 32]:
            x = Conv2DTranspose(f, (3, 3), strides=2, padding="same")(x)
            x = LeakyReLU()(x)

        x = Conv2DTranspose(3, (3, 3), strides=2, padding="same")(x)
        outputs = Activation("sigmoid")(x)
        self.decoder_model = Model(latentInputs, outputs, name="decoder")
        self.autoencoder_model = Model(inputs, self.decoder_model(self.encoder_model(inputs)),name="autoencoder")
        
        return None
    
    def get_encoder(self):
        if self.encoder_model is not None:
            return self.encoder_model
        else:
            print("Encoder model has not been defined!")
            return None

    def get_decoder(self):
        if self.decoder_model is not None:
            return self.decoder_model
        else:
            print("Decoder model has not been defined!")
            return None
    
    def get_autoencoder(self):
        if self.autoencoder_model is not None:
            return self.autoencoder_model
        else:
            print("Autoencoder model has not been defined!")
            return None


model = ConvAutoencoder()
model.build()
encoder = model.get_encoder()
decoder = model.get_decoder()
autoencoder = model.get_autoencoder()
encoder.summary()
decoder.summary()
autoencoder.summary()


#LEARNING_RATE = 0.0005
#N_EPOCHS = 20

#optimizer = Adam(lr = LEARNING_RATE)

#def r_loss(y_true, y_pred):
 #   return K.mean(K.square(y_true - y_pred), axis = [1,2,3])

#autoencoder.compile(optimizer=optimizer, loss = r_loss)
#
#checkpoint_ae_best = ModelCheckpoint(os.path.join(WEIGHTS_FOLDER, 'AE/autoencoder_best_weights.h5'), 
#                                     monitor='val_loss',
 #                                    mode='min',
 #                                    save_best_only=True,
 #                                    save_weights_only = False, 
 #                                    verbose=1)
#
#checkpoint_ae_last = ModelCheckpoint(os.path.join(WEIGHTS_FOLDER, 'AE/autoencoder_last_weights.h5'), 
 #                                    monitor='val_loss',
 #                                    mode='min',
 #                                    save_best_only=True,
 #                                    save_weights_only = False, 
 #                                    verbose=1)


#autoencoder.fit(train_dataset,
#                epochs=10,
#                callbacks=[checkpoint_ae_best, checkpoint_ae_last],
#                validation_data=valid_dataset)


test_dataset = build_dataset(valid_paths, test=True)
autoencoder.load_weights('./weights/AE/autoencoder_last_weights.h5')
#data = list(test_dataset.take(20))
#
#fig = plt.figure(figsize=(30, 10))
#for n in range(0, 20, 2):
#    image = autoencoder.predict(data[n])
#    
#    plt.subplot(2, 10, n + 1)
#    plt.imshow(np.squeeze(data[n]))
#    plt.title('original image')
#    
#    plt.subplot(2, 10, n + 2)
#    plt.imshow(np.squeeze(image))
#    plt.title('reconstruct')
#    
#plt.show()


data = list(test_dataset.take(20))

fig = plt.figure(figsize=(30, 10))
for n in range(0, 20, 2):
    image = encoder.predict(data[n])
    image += np.random.normal(0.0, 1.0, size = (Z_DIM)) #inject noise to the encoded vector
    reconst_images = decoder.predict(image)
    
    plt.subplot(2, 10, n + 1)
    plt.imshow(np.squeeze(data[n]))
    plt.title('original image')
    
    plt.subplot(2, 10, n + 2)
    plt.imshow(np.squeeze(reconst_images))
    plt.title('Reconstruct noisy image')
    
plt.show()













































