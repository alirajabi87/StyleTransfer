# Style Transfer Leaning
import os

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten, AveragePooling2D, MaxPooling2D, Conv2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.preprocessing import image

import tensorflow.keras.backend as K
import numpy as np
from PIL import Image as PL
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from StyleContentModel import StyleContentModel


mpl.rcParams['figure.figsize'] = (12, 12)
mpl.rcParams['axes.grid'] = False

from scipy.optimize import fmin_l_bfgs_b
import functools
from datetime import datetime


def VGG16_AvgPool(shape):
    vgg = VGG16(input_shape=shape, weights='imagenet', include_top=False)

    new_model = Sequential()
    for layer in vgg.layers:
        if layer.__class__ == MaxPooling2D:
            # Replace Maxpooling2D with AvgPooling2D
            new_model.add(AveragePooling2D())
        else:
            new_model.add(layer)

    # print(new_model.summary())
    return new_model


def VGG16_AvgPool_Cutoff(shape, num_conv):
    if num_conv < 1 or num_conv > 13:
        print("Number of Convelution layers should be in range [1, 13]")
        return None
    model = VGG16_AvgPool(shape)
    new_model = Sequential()
    n = 0
    for layer in model.layers:
        if layer.__class__ == Conv2D:
            # print('one Conv2D has been added')
            n += 1
        new_model.add(layer)
        # print(layer.__class__)
        if n >= num_conv:
            break
    # print(new_model.summary())
    return model


def unpreprocess(img):
    img[..., 0] += 103.039
    img[..., 1] += 116.779
    img[..., 2] += 126.68
    img = img[..., ::-1]
    return img


def scale_img(x):
    x -= x.min()
    x /= x.max()
    return x


def tensor_to_image(tensor):
    tensor += 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PL.fromarray(tensor)


def load_img(path):
    max_dim = 512
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)
    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img


def imshow(image, title=None):
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)
        print("image has been reduce to 3 channel")

    plt.imshow(image)
    if title:
        plt.title(title)


def vgg_layers(layer_names):
    red_model = Sequential()
    vgg = tf.keras.applications.vgg16.VGG16(include_top=False, weights='imagenet')
    vgg.trainable = False
    for layer in vgg.layers:
        if layer.__class__ == MaxPooling2D:
            red_model.add(AveragePooling2D())
        else:
            red_model.add(layer)
    outputs = [red_model.get_layer(name).output for name in layer_names]
    model = Model([red_model.input], outputs)
    return model


if __name__ == '__main__':
    content_path = "../DATA/YellowLabradorLooking_new.jpg"
    style_path = "../DATA/Vassily_Kandinsky,_1913_-_Composition_7.jpg"

    content_image = load_img(content_path)
    style_image = load_img(style_path)

    plt.subplot(1, 2, 1)
    imshow(content_image, 'Content Image')

    plt.subplot(1, 2, 2)
    imshow(style_image, 'Style Image')
    plt.show()

    # Checking the dimensions
    print(
        f"Content image max: {tf.math.reduce_max(content_image)}, content image min: {tf.math.reduce_min(content_image)}")
    print(f"Style image max: {tf.math.reduce_max(style_image)}, Style image min: {tf.math.reduce_min(style_image)}")

    # Test VGG16 for the Content image
    x = tf.keras.applications.vgg16.preprocess_input(content_image * 255)
    x = tf.image.resize(x, (224, 224))
    vgg = tf.keras.applications.vgg16.VGG16(include_top=True, weights='imagenet')
    pred = vgg(x)
    # print(f" prediction probability: {pred}, Shape: {pred.shape}")

    prediction_top_5 = tf.keras.applications.vgg16.decode_predictions(pred.numpy(), top=5)[0]

    list_pred = [(class_name, prob) for (number, class_name, prob) in prediction_top_5]
    print(list_pred)

    shape = (224, 224, 3)
    print(content_image.shape)
    vgg = VGG16_AvgPool(content_image.shape[1:])
    print(vgg.summary())

    content_layers = ['block5_conv2']
    style_leyers = ['block1_conv2',
                    'block2_conv2',
                    'block3_conv2',
                    'block4_conv2',
                    'block5_conv2']
    num_content_layers = len(content_layers)
    num_style_leyers = len(style_leyers)

    style_extractor = vgg_layers(style_leyers)
    style_output = style_extractor(style_image*255)

    for name, output in zip(style_leyers, style_output):
        print(name)
        print(f" Shape: {output.numpy().shape }")
        print(f" min: {output.numpy().min() }")
        print(f" max: {output.numpy().max() }")
        print(f" mean: {output.numpy().mean() }")
        print("-------------------------------------------")
    extractor = StyleContentModel(style_leyers, content_layers, vgg_layers)
    results = extractor(tf.constant(content_image))
    print("Style: ")
    for name, output in sorted(results['style'].items()):
        print("  ", name)
        print("    shape: ", output.numpy().shape)
        print("    min: ", output.numpy().min())
        print("    max: ", output.numpy().max())
        print("    mean: ", output.numpy().mean())
        print()

    print("Contents:")
    for name, output in sorted(results['content'].items()):
        print("  ", name)
        print("    shape: ", output.numpy().shape)
        print("    min: ", output.numpy().min())
        print("    max: ", output.numpy().max())
        print("    mean: ", output.numpy().mean())