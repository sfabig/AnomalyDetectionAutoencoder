import numpy as np
import keras as ke
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img, img_to_array, save_img
from keras.layers import Input, Dense, Lambda, Flatten, Reshape, Layer
from keras.layers import Conv2D, Conv2DTranspose
from keras.models import Model
import os
import argparse
from keras import backend as K

import csv

seed = 11037

curdir = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser()
parser.add_argument("--data", choices=['bottle','carpet'], default='bottle')
parser.add_argument('--optimizer', choices=['adam','sgd','adagrad','rmsprop'], default='adam')
parser.add_argument('--loss', choices=['mean_squared_error', 'binary_crossentropy'], default='binary_crossentropy')
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--test_samples', type=float, default=0.2)
parser.add_argument('--training', choices=[True,False],default=False)
parser.add_argument('--saveweights', choices=[True,False],default=True)
parser.add_argument('--predict', choices=[True,False],default=True)

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

def build_model_enc(input_shape, kernel_size, filters, latent_dim):
    # build VAE encoder model
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
    latent_shape = K.int_shape(x)
    
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
    
    return encoder, latent_shape

def build_model_dec(latent_shape, latent_dim, filters, kernel_size):
    # build decoder model
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    x = Dense(latent_shape[1] * latent_shape[2] * latent_shape[3], activation='relu')(latent_inputs)
    x = Reshape((latent_shape[1], latent_shape[2], latent_shape[3]))(x)
    
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

    return decoder

def build_model_disc(latent_dim):
    model = ke.models.Sequential()
    model.add(ke.layers.Dense(32, activation="relu", input_shape=(latent_dim,)))
    model.add(ke.layers.Dense(32, activation="relu"))
    model.add(ke.layers.Dense(1, activation="sigmoid"))
    return model

def build_model_aae(input_shape, kernel_size, filters, latent_dim):
    model_enc, latent_shape = build_model_enc(input_shape, kernel_size, filters, latent_dim)
    model_dec = build_model_dec(latent_shape, latent_dim, filters, kernel_size)
    model_disc = build_model_disc(latent_dim)
    
    model_ae = Model(model_enc.inputs, model_dec(model_enc(model_enc.inputs)[2]))
    
    model_enc_disc = Model(model_enc.inputs, model_disc(model_enc(model_enc.inputs)[2]))
    
    return model_enc, model_dec, model_disc, model_ae, model_enc_disc
            
def settrainable(model, toset):
    for layer in model.layers:
        layer.trainable = toset
    model.trainable = toset
    
def main(args):
    input_shape = (224, 224, 3)
    batch_size = 32
    kernel_size = 3
    filters = 16
    latent_dim = 2
    epochs = 50
    do_train = False
    data = 'carpet'
    optimizer='rmsprop'
    loss = 'binary_crossentropy'
    
    model_enc, model_dec, model_disc, model_ae, model_enc_disc = build_model_aae(input_shape, kernel_size, filters, latent_dim)
    
    model_enc.summary()
    model_dec.summary()
    model_disc.summary()
    model_ae.summary()
    model_enc_disc.summary()
    
    model_disc.compile(optimizer=optimizer, loss=loss)
    model_enc_disc.compile(optimizer=optimizer, loss=loss)
    model_ae.compile(optimizer=optimizer, loss=loss)
    
    #load data
    x_train = load_data(data, (224,224))
    x_train = x_train.astype('float32') / 255.0 

    if(do_train):
        
        #Set Number of Epochs to 10-20 or higher.
        history_rec_loss = []
        history_adv_loss = []
        for epochnumber in range(50):
            print('Epoch ' + str(epochnumber+1))
            np.random.shuffle(x_train)
            
            for i in range(int(len(x_train) / batch_size)):
                settrainable(model_ae, True)
                settrainable(model_enc, True)
                settrainable(model_dec, True)
                batch = x_train[i*batch_size:i*batch_size+batch_size]
                print('Train AE Model')
                model_ae.train_on_batch(batch, batch)
                
                settrainable(model_disc, True)
                batchpred = model_enc.predict(batch)[-1]
                fakepred = np.random.standard_normal((batch_size,latent_dim))
                discbatch_x = np.concatenate([batchpred, fakepred])
                discbatch_y = np.concatenate([np.zeros(batch_size), np.ones(batch_size)])
                print('Train Disc Model')
                model_disc.train_on_batch(discbatch_x, discbatch_y)
                
                settrainable(model_enc_disc, True)
                settrainable(model_enc, True)
                settrainable(model_disc, False)
                print('Train Enc-Disc Model')
                model_enc_disc.train_on_batch(batch, np.ones(batch_size))
                
            rec_loss = model_ae.evaluate(x_train, x_train, verbose=0)
            adv_loss = model_enc_disc.evaluate(x_train, np.ones(len(x_train)), verbose=0)
            history_rec_loss.append(rec_loss)
            history_adv_loss.append(adv_loss)
            print ("Reconstruction Loss:", rec_loss)
            print ("Adverserial Loss:", adv_loss)
            
        plt.plot(history_rec_loss, label = 'Reconstruction Loss')
        plt.plot(history_adv_loss, label='Adverserial Loss')
        plt.legend()
        plt.savefig('./images/aae/aae_rec_adv_loss_' + data + '_' + str(epochs) + '_' + loss + '_' + optimizer + '.png')
            
        print('Saving Model...')
        model_ae.save_weights('./models/aae/aae_enc_dec_'+ data + '_' + str(epochs) + '_' + loss + '_' + optimizer + '_weights.h5')
        model_enc_disc.save_weights('./models/aae/aae_enc_disc_'+ data + '_' + str(epochs) + '_' + loss + '_' + optimizer + '_weights.h5')
        model_disc.save_weights('./models/aae/aae_disc_'+ data + '_' + str(epochs) + '_' + loss + '_' + optimizer + '_weights.h5')
    else:
        print('Loading Model weights...')
        model_ae.load_weights('./models/aae/aae_enc_dec_'+ data + '_' + str(epochs) + '_' + loss + '_' + optimizer + '_weights.h5')
        model_enc_disc.load_weights('./models/aae/aae_enc_disc_'+ data + '_' + str(epochs) + '_' + loss + '_' + optimizer + '_weights.h5')
        model_disc.load_weights('./models/aae/aae_disc_'+ data + '_' + str(epochs) + '_' + loss + '_' + optimizer + '_weights.h5')
        print('Done')
        
    print('Predicting...')
    csv_name='losses.csv'
    max_error = model_ae.evaluate(x_train,x_train,batch_size=batch_size)
    test_directory = './' + data + '/test/'
    result_directory = './results/AAE/' + data + '/E' + str(epochs) + '_' + loss + '_' + optimizer + '/'
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
            prediction = model_ae.predict(img)
            predict_name = filename + '_predict.png'
            save_img(os.path.join(result_directory,dir,predict_name),prediction[0])
            this_error = model_ae.evaluate(img,img)
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
    accuracy_file = open(os.path.join(result_directory,"accuracy.txt"), "w")
    accuracy_file.write(str(accuracy/element_count))
    accuracy_file.close()
        
if __name__ == '__main__':
    args = parser.parse_args()
    main(args)