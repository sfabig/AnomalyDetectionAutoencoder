from keras.preprocessing.image import load_img, img_to_array, save_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, UpSampling2D
import numpy as np
import os
import argparse
import csv
import matplotlib.pyplot as plt

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

def load_model():
    input_shape=(224,224,3)
    n_channels = input_shape[-1]
    model = Sequential()
    model.add(Conv2D(32, (3,3), activation='relu', padding='same', input_shape=input_shape))
    model.add(MaxPool2D(padding='same'))
    model.add(Conv2D(16, (3,3), activation='relu', padding='same'))
    model.add(MaxPool2D(padding='same'))
    model.add(Conv2D(8, (3,3), activation='relu', padding='same'))
    model.add(UpSampling2D())
    model.add(Conv2D(16, (3,3), activation='relu', padding='same'))
    model.add(UpSampling2D())
    model.add(Conv2D(32, (3,3), activation='relu', padding='same'))
    model.add(Conv2D(n_channels, (3,3), activation='sigmoid', padding='same'))
    model.compile(optimizer=args.optimizer, loss=args.loss)
    return model


def main(args):
    # instantiate model
    model = load_model()
    
    train = load_data(args.data, (224,224))
    train = train.astype('float32') / 255.0      
    
    model.summary()
    if(args.training):
        print('Training...')
        model.fit(x=train, y=train, batch_size=args.batch_size, epochs=args.epochs, validation_split=args.test_samples)
        if(args.saveweights):
            print('Saving Model...')
            model.save_weights('./models/cae/cae_'+ args.data + '_' + str(args.epochs) + '_' + args.loss + '_' + args.optimizer + '_weights.h5')
    
        plt.plot(model.history.history['loss'], label = 'loss')
        plt.plot(model.history.history['val_loss'], label='val_loss')
        plt.legend()
        plt.savefig('./images/cae/cae_loss_' + args.data + '_' + str(args.epochs) + '_' + args.loss + '_' + args.optimizer + '.png')
    else:
        print('Loading weights…')
        model.load_weights('./models/cae/cae_'+ args.data + '_' + str(args.epochs) + '_' + args.loss + '_' + args.optimizer + '_weights.h5')
        print('Done')
    
    if(args.predict):
        print('Predicting...')
        csv_name='losses.csv'
        max_error = model.evaluate(train,train,batch_size=args.batch_size)
        test_directory = './' + args.data + '/test/'
        result_directory = './results/CAE/' + args.data + '/E' + str(args.epochs) + '_' + args.loss + '_' + args.optimizer + '/'
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
                prediction = model.predict(img)
                predict_name = filename + '_predict.png'
                save_img(os.path.join(result_directory,dir,predict_name),prediction[0])
                this_error = model.evaluate(img,img)
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
    
    del(model)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
