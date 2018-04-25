"""
This file contains methods to read various datasets.
"""

import os

import tensorflow as tf
import cv2
import numpy as np

FLAGS = tf.app.flags.FLAGS

MNIST_TRAIN_IMAGE = ['../datasets/mnist/train-images-idx3-ubyte/data']
MNIST_TRAIN_LABEL = ['../datasets/mnist/train-labels-idx1-ubyte/data']

MNIST_TEST_IMAGE = ['../datasets/mnist/t10k-images-idx3-ubyte/data']
MNIST_TEST_LABEL = ['../datasets/mnist/t10k-labels-idx1-ubyte/data']

CALTECH_101_DIR = '../datasets/caltech_101/101_ObjectCategories'


def mnist_input(train):
    """
    train: flag indicating train images or test images
    """

    if train:
        filename_image_list = MNIST_TRAIN_IMAGE
        filename_label_list = MNIST_TRAIN_LABEL
    else:
        filename_image_list = MNIST_TEST_IMAGE
        filename_label_list = MNIST_TEST_LABEL

    label_bytes = 1
    height = 28
    width = 28
    depth = 1

    image_bytes = height * width * depth

    image_dataset = tf.data.FixedLengthRecordDataset(filename_image_list, 
                                                     record_bytes=image_bytes, 
                                                     header_bytes=16)
    label_dataset = tf.data.FixedLengthRecordDataset(filename_label_list, 
                                                     record_bytes=label_bytes, 
                                                     header_bytes=8)

    image_dataset = image_dataset.map(lambda x: 
                                      tf.decode_raw(x, tf.uint8)).map(lambda x: 
                                                                      tf.cast(x, tf.float32))
    image_dataset = image_dataset.map(lambda x: 
                                      tf.reshape(x, [depth, width, height]))
    image_dataset = image_dataset.map(lambda x: tf.transpose(x, [2, 1, 0]))

    label_dataset = label_dataset.map(lambda x: 
                                      tf.decode_raw(x, tf.uint8)).map(lambda x: 
                                                                      tf.cast(x, tf.int32))

    label_dataset = label_dataset.map(lambda x: x[0])
    # label_dataset = label_dataset.map(lambda x: tf.one_hot(x[0], depth=10)) # was x[0]

    final_dataset = tf.data.Dataset.zip((image_dataset, label_dataset))

    if train:
        final_dataset = final_dataset.repeat().batch(FLAGS.batch_size)
    else:
        final_dataset = final_dataset.batch(FLAGS.batch_size)

    final_dataset_iterator = final_dataset.make_one_shot_iterator()
    return final_dataset_iterator.get_next()


def caltech_101_get_image_label(train):
    """
    train: True if the train dataset is required
    """
    classes = [name for name in os.listdir(CALTECH_101_DIR) if 
               os.path.isdir(os.path.join(CALTECH_101_DIR, name))]
    classes = sorted(classes)

    files_s = []

    for folder in classes:
        parent = os.path.join(CALTECH_101_DIR, folder)
        files = [os.path.join(parent, name) for name in os.listdir(parent) if 
                 os.path.isfile(os.path.join(parent, name))]
        files = sorted(files)
        files_s.append(files)

    # print(files_s)

    # either sort file names and choose first 30 or choose 30 files randomly 
    image_files = []
    labels = []
    for i, files in enumerate(files_s):
        # i is the index of class to which files in files belong

        # pick 30 files
        if train:
            picked_files = files[:30]
        else:
            picked_files = np.random.choice(files, 30)

        # for file in picked_files:
        #     image = cv2.imread(file, 0)
        #     image = np.reshape(image, [image.shape[0], image.shape[1], 1])
        #     yield (image, i)

        for file in picked_files:
            # image = tf.image.decode_jpeg(file, channels=1)
            image_files.append(file)
            labels.append(i)

    return tf.constant(image_files), tf.constant(labels)


def input_parser(img_path, label):

    img_file = tf.read_file(img_path)
    image = tf.image.decode_jpeg(img_file, channels=1)
    image = tf.image.resize_image_with_crop_or_pad(image, 151, 151)
    image = tf.image.per_image_standardization(image)

    return image, label


def caltech_101_input(train):
    """
    train: True if the train dataset is required
    """

    # dataset = tf.data.Dataset.from_generator(lambda : caltech_101_get_image_label(train), 
    #                                          (tf.uint8, tf.int32),
    #                                          (tf.TensorShape([None, None, 1]), tf.TensorShape([])))
    # dataset = dataset.map(lambda x, y: (tf.cast(x, tf.float32), y))
    # dataset = dataset.map(lambda x, y: (tf.image.resize_image_with_crop_or_pad(x, 151, 151), y))
    # dataset = dataset.map(lambda x, y: (tf.image.per_image_standardization(x), y))

    images, labels = caltech_101_get_image_label(train)
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    # dataset = dataset.map(input_parser, 8)
    dataset = dataset.map(lambda x, y: (tf.read_file(x), y), 8)
    dataset = dataset.map(lambda x, y: (tf.image.decode_jpeg(x, channels=1), y), 8)
    dataset = dataset.map(lambda x, y: (tf.image.resize_image_with_crop_or_pad(x, 151, 151), y), 8)
    dataset = dataset.map(lambda x, y: (tf.image.per_image_standardization(x), y), 8)

    dataset = dataset.prefetch(3*FLAGS.batch_size)

    if train:
        dataset = dataset.repeat().shuffle(tf.shape(images, out_type=tf.int64)[0]).batch(FLAGS.batch_size)
    else:
        dataset = dataset.batch(FLAGS.batch_size)

    return dataset.make_one_shot_iterator().get_next()