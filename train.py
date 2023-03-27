import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import pandas as pd

from tensorflow.keras.preprocessing import image_dataset_from_directory

import argparse


train_dir = '../dataset/train'
val_dir = '../dataset/val'
batch_size = 16
image_size = (224, 224)

lr = 0.001
momentum = 0.9
epoch = 100

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='MobileNetV2')
    parser.add_argument('--aug', type=str)
    parser.add_argument('--opt', type=str)
    parser.add_argument('--freeze', type=str)

    return parser.parse_args()

def run_training():
    args = parse_args()
    model_name = args.model + '_aug_' + args.aug + '_opt_' + args.opt + '_freeze_' + args.freeze
    print(model_name)
    train_dataset = image_dataset_from_directory(train_dir,
                                                 shuffle=True,
                                                 batch_size=batch_size,
                                                 image_size=image_size)

    val_dataset = image_dataset_from_directory(val_dir,
                                                 shuffle=False,
                                                 batch_size=batch_size,
                                                 image_size=image_size)
    if args.model == 'MobileNetV2' :
        from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
        basemodel = MobileNetV2(input_shape=(224,224,3), include_top=False, weights='imagenet')
    elif args.model == 'DenseNet121' :
        from tensorflow.keras.applications.densenet import DenseNet121, preprocess_input
        basemodel = DenseNet121(input_shape=(224,224,3), include_top=False, weights='imagenet')
    else:
        from tensorflow.keras.applications.resnet import ResNet50, preprocess_input
        basemodel = ResNet50(input_shape=(224,224,3), include_top=False, weights='imagenet')

    data_augmentation = tf.keras.Sequential([
          tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal_and_vertical'),
          tf.keras.layers.experimental.preprocessing.RandomRotation(0.2)])
    
    inputs = tf.keras.Input(shape=(224, 224, 3))
    if args.aug == 'True' :
        x = data_augmentation(inputs)
        
    
    x = preprocess_input(inputs)



    x = basemodel(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)


    model = tf.keras.Model(inputs, outputs)
    
    if args.freeze == 'True':
        for layers in model.layers[:-1]:
            layers.trainable = False
    else :
        for layers in model.layers:
            layers.trainable = True

    if args.opt == 'sgd' :
        opt = tf.keras.optimizers.SGD(lr=lr, momentum=momentum)
    else :
        opt = tf.keras.optimizers.Adam(lr=lr)
        
    model.compile(optimizer = opt, loss='binary_crossentropy', metrics=['accuracy'])

    model.summary()

    history = model.fit(train_dataset,
                        epochs=epoch,
                        validation_data=val_dataset)

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']


    fig, loss_ax = plt.subplots()
    acc_ax = loss_ax.twinx()

    loss_ax.plot(loss, 'y', label='train loss')
    loss_ax.plot(val_loss, 'r', label='val loss')
    acc_ax.plot(acc, 'b', label='train acc')
    acc_ax.plot(val_acc, 'g', label='val acc')


    loss_ax.set_xlabel('epoch')
    loss_ax.set_ylabel('loss')
    acc_ax.set_ylabel('accuray')

    loss_ax.legend(loc='upper left')
    acc_ax.legend(loc='lower left')

    plt.savefig('../result/'+model_name+'.png')

    result = pd.DataFrame(history.history)
    result.to_csv('../result/'+model_name+'.csv')
    
if __name__=="__main__":
    run_training()