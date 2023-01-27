import os
import pickle
import numpy as np
import pandas as pd
from zipfile import ZipFile
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import load_img, img_to_array, array_to_img
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Input, Lambda



# main function to unzip, resize and transform images
def get_features():

    img_dir = './food'
    if not os.path.exists(img_dir):
        # Extract files from food zip
        zip_ref = ZipFile('food.zip', 'r')
        zip_ref.extractall()
        img_dir = zip_ref.filename[:-4]
        zip_ref.close()

    # get image resized from new food file
    res_img_dir = './resized_images'
    if not os.path.exists(res_img_dir):
        os.makedirs(res_img_dir)
        num = 0
        image_dimension = [299, 299]

        for filename in os.listdir(img_dir):
            num += 1
            print('Image extracted: ', num, end="\r")
            if filename.endswith('.jpg'):
                image = img_to_array(load_img(img_dir + '/' + filename))
                # resize tf way, convert back to image and save
                image = array_to_img(tf.image.resize_with_pad(image, image_dimension[0], image_dimension[1], antialias=True))
                image.save(res_img_dir + '/' + str(int(os.path.splitext(filename)[0])) + '.jpg')

        print('Extraction terminated')

    feat_pickle = 'features_resnet.pckl'
    if not os.path.exists(feat_pickle):

        images_resized = image_loader(res_img_dir, 1)

        # our backbone requires the image dimension to be fixed
        model = backbone((299, 299, 3), backbone="InceptionResNetV2")
        # using dynamic shapes for backbone degrade results

        print("Feature extraction")
        features = model.predict(images_resized, steps=10000)

        with open(feat_pickle, 'wb') as f:
            pickle.dump(features, f)

        print("Feature extraction and loading terminated")

    else:
        with open(feat_pickle, 'rb') as f:
            features = pickle.load(f)
        print("Current existing feature file ", feat_pickle, " has been loaded")

    return features

# backbone of our network to extract features
def backbone(input_shape, backbone : str):

    # feature extraction
    if backbone == "InceptionResNetV2":
        backbone_model = tf.keras.applications.InceptionResNetV2(pooling='avg', include_top=False)
    if backbone == "VGG16":
        backbone_model = tf.keras.applications.VGG16(include_top=False)
    if backbone == "InceptionV3":
        backbone_model = tf.keras.applications.InceptionV3(pooling='avg', include_top=False)

    backbone_model.trainable = False

    x_in = Input(shape=input_shape)
    x = backbone_model(x_in)
    model = Model(inputs=x_in, outputs=x)

    return model

# Image generator, resize and load images
def image_loader(folder, batch_dim):

    img_id = 0
    while True:
        batch_vec = []

        while len(batch_vec) < batch_dim:
            img_name = folder + "/" + str(int(img_id)) + ".jpg"
            img = load_img(img_name)
            img = tf.keras.applications.inception_resnet_v2.preprocess_input(img_to_array(img))
            batch_vec.append(img)
            img_id = (img_id + 1) % 10000

        batch_vec = np.array(batch_vec)
        label_vec = np.zeros(batch_dim)

        try:
            yield batch_vec, label_vec
        except StopIteration:
            return


def classificator_model(input_tens):
    input_x = Input(input_tens.shape[1:])
    x = Activation('leaky_relu')(input_x)
    x = Dropout(0.76)(x)
    x = Dense(1000)(x)
    x = Activation('leaky_relu')(x)
    x = Dense(200)(x)
    x = Activation('leaky_relu')(x)
    x = Dropout(0.74)(x)
    x = Dense(100)(x)
    x = Activation('leaky_relu')(x)
    x = Dense(20)(x)
    x = Activation('leaky_relu')(x)
    x = Dense(1)(x)
    x = Activation('sigmoid')(x)
    model = Model(inputs=input_x, outputs=x)
    return model


def get_triplet_dataset(triplet_list, features, lab=False):

    df = pd.read_csv(triplet_list, delim_whitespace=True, header=None, names=["A", "B", "C"])
    trip_tot = len(df)

    training_vec = []
    label_vec = []

    for i in range(trip_tot):

        trip = df.iloc[i]

        A, B, C = trip['A'], trip['B'], trip['C']
        tensor_a, tensor_b, tensor_c = features[A], features[B], features[C]
        triplet_vec = np.concatenate((tensor_a, tensor_b, tensor_c), axis=-1)

        if lab:
            triplet_vec_inverse = np.concatenate((tensor_a, tensor_c, tensor_b), axis=-1)
            training_vec.append(triplet_vec)
            label_vec.append(1)
            training_vec.append(triplet_vec_inverse)
            label_vec.append(0)

        else:
            training_vec.append(triplet_vec)

    training_vec = np.array(training_vec)

    if lab:
        label_vec = np.array(label_vec)
        return training_vec, label_vec

    else:
        return training_vec
