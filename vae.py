import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from keras.preprocessing.image import load_img, img_to_array, array_to_img, save_img
from keras.layers import Input, Dense, Lambda, Flatten, Reshape, Layer
from keras.layers import Conv2D, Conv2DTranspose
from keras.models import Model
from keras import backend as K
from keras import metrics
from keras.utils import plot_model

from keras.losses import mse, binary_crossentropy
import os
import csv

def load_data(data_set, target_size=None):
    images = []
    directory = './' + data_set + '/train/good/'
    for filename in os.listdir(directory):
        img = load_img(os.path.join(directory,filename), target_size = target_size)
        img = img_to_array(img)
        images.append(img)
    images = np.stack(images)
    return images
    
# reparameterization trick
# instead of sampling from Q(z|X), sample eps = N(0,I)
# then z = z_mean + sqrt(var)*eps
def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.
    # Arguments
        args (tensor): mean and log of variance of Q(z|X)
    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

input_shape = (224, 224, 3)
batch_size = 32
kernel_size = 3
filters = 16
latent_dim = 2
epochs = 50
do_train = True
data = 'carpet'
optimizer='rmsprop'
loss = 'mean_squared_error'

# VAE model = encoder + decoder
# build encoder model
inputs = Input(shape=input_shape, name='encoder_input')
x = inputs
for i in range(2):
    filters *= 2
    x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               activation='relu',
               strides=2,
               padding='same')(x)

# shape info needed to build decoder model
shape = K.int_shape(x)

# generate latent vector Q(z|X)
x = Flatten()(x)
x = Dense(16, activation='relu')(x)
z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)

# use reparameterization trick to push the sampling out as input
# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

# instantiate encoder model
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
encoder.summary()
#plot_model(encoder, to_file='vae_cnn_encoder.png', show_shapes=True)

# build decoder model
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
x = Dense(shape[1] * shape[2] * shape[3], activation='relu')(latent_inputs)
x = Reshape((shape[1], shape[2], shape[3]))(x)

for i in range(2):
    x = Conv2DTranspose(filters=filters,
                        kernel_size=kernel_size,
                        activation='relu',
                        strides=2,
                        padding='same')(x)
    filters //= 2

outputs = Conv2DTranspose(filters=3,
                          kernel_size=kernel_size,
                          activation='sigmoid',
                          padding='same',
                          name='decoder_output')(x)

# instantiate decoder model
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()
#plot_model(decoder, to_file='vae_cnn_decoder.png', show_shapes=True)

# instantiate VAE model
outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='vae')


train = load_data(data, (224,224))
train = train.astype('float32') / 255.0

image_size = train.shape[1]

# VAE loss = mse_loss or xent_loss + kl_loss
def vae_loss(x, x_decoded_mean_squash):
    if loss == 'mean_squared_error':
        reconstruction_loss = mse(K.flatten(x), K.flatten(x_decoded_mean_squash))
    elif loss == 'binary_crossentropy':
        reconstruction_loss = binary_crossentropy(K.flatten(x),
                                                  K.flatten(x_decoded_mean_squash))
    
    reconstruction_loss *= image_size * image_size
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    return K.mean(reconstruction_loss + kl_loss)
    
#vae.add_loss(vae_loss)
vae.compile(optimizer=optimizer, loss=vae_loss)
vae.summary()

if(do_train):
    vae.fit(x=train, y=train, batch_size=64, epochs=50, validation_split=0.2)
    
    plt.plot(vae.history.history['loss'], label = 'loss')
    plt.plot(vae.history.history['val_loss'], label='val_loss')
    plt.legend()
    plt.savefig('./images/vae/vae_loss_' + data + '_' + str(epochs) + '_kld+' + loss + '_' + optimizer + '.png')

    print('Saving Model...')
    vae.save_weights('./models/vae/vae_'+ data + '_' + str(epochs) + '_kld+' + loss + '_' + optimizer + '_weights.h5')
else:
    print('Loading weights…')
    vae.load_weights('./models/vae/vae_'+ data + '_' + str(epochs) + '_kld+' + loss + '_' + optimizer + '_weights.h5')
    print('Done')
    

print('Predicting...')
csv_name='losses.csv'
max_error = vae.evaluate(train,train,batch_size=batch_size)
test_directory = './' + data + '/test/'
result_directory = './results/VAE/' + data + '/E' + str(epochs) + '_kld+' + loss + '_' + optimizer + '/'
if not os.path.exists(result_directory):
    os.makedirs(result_directory)

anomaly_list=list()
for dir in os.listdir(test_directory):
    print(dir)
    mse_list=list()
    dir_list = list()
    dir_list.append(dir)
    if not os.path.exists(os.path.join(result_directory,dir)):     
        os.mkdir(os.path.join(result_directory,dir))
        
    for filename in os.listdir(os.path.join(test_directory,dir)):
        img = load_img(os.path.join(test_directory,dir,filename), target_size = (224,224))
        img = img_to_array(img)
        img = np.expand_dims(img,axis=0)
        img = img.astype('float32') / 255.0
        prediction = vae.predict(img)
        predict_name = filename + '_predict.png'
        save_img(os.path.join(result_directory,dir,predict_name),prediction[0])
        this_error = vae.evaluate(img,img)
        mse_list.append(this_error)
        print('This error:' + str(this_error) + ', Max Error:' + str(max_error + max_error*0.05))
        
        if(this_error < max_error + max_error*0.05):
            dir_list.append(False)
        else:
            dir_list.append(True)
            
    dir_accuracy = 0
    if 'good' in dir_list:
        dir_accuracy += dir_list.count(False)/(len(dir_list)-1)
    else:
        dir_accuracy += dir_list.count(True)/(len(dir_list)-1)
    dir_accuracy_file = open(os.path.join(result_directory,dir,"accuracy.txt"), "w")
    dir_accuracy_file.write(str(dir_accuracy))
    dir_accuracy_file.close()
    
    anomaly_list.append(dir_list)
    
    with open(os.path.join(result_directory,dir,csv_name), 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(mse_list)
# calculate accuracy
accuracy = 0
element_count = 0
for i in range(len(anomaly_list)):
    element_count += len(anomaly_list[i]) - 1
    if 'good' in anomaly_list[i]:
        accuracy += anomaly_list[i].count(False)
    else:
        accuracy += anomaly_list[i].count(True)
print('Anomaly Detection Accuracy: ' + str(accuracy/element_count*100) + '%')
text_file = open(os.path.join(result_directory,"accuracy.txt"), "w")
text_file.write(str(accuracy/element_count))
text_file.close()