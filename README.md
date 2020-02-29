# AnomalyDetectionAutoencoder
Project Repository exploring various Autoencoders for Anomaly Detection

Trains Convolutional, Variational and Adversarial Autoencoders for Anomaly Detection in MvTEC Datasets "bottle" and "carpet"

## Prerequisites
```
Keras 2.2.4
Tensorflow 1.14.0
```
Used Data for Training and Evaluation from [here](https://www.mvtec.com/de/unternehmen/forschung/datasets/mvtec-ad/)
```
Images should be organized as follows:
"./name_of_dataset/train/" for Training data 
and 
"./name_of_dataset/train/" for Test data
```

## Running
Executing any .py with datasets present in the given structure trains the respective model and evaluates it on the data in /test/
Evaluation reconstructs test images in "./result/used_model/name_of_dataset/model_configuration/" and calculates accuracy over all images and every subfolder respectively
