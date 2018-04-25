"""
This file contain functions to read dataset.
"""

import tensorflow as tf
import pandas as pd
import numpy as np


TRAIN_URL = "http://download.tensorflow.org/data/iris_training.csv"
TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"

CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth',
                    'PetalLength', 'PetalWidth', 'Species']

SPECIES = ['Setosa', 'Versicolor', 'Virginica']

mnist_train_filename_image_list = ["./datasets/train-images-idx3-ubyte/data"]
mnist_train_filename_label_list = ["./datasets/train-labels-idx1-ubyte/data"]

mnist_eval_filename_image_list = ["./datasets/t10k-images-idx3-ubyte/data"]
mnist_eval_filename_label_list = ["./datasets/t10k-labels-idx1-ubyte/data"]


def maybe_download():
    train_path = tf.keras.utils.get_file(TRAIN_URL.split('/')[-1], TRAIN_URL)
    test_path = tf.keras.utils.get_file(TEST_URL.split('/')[-1], TEST_URL)

    return train_path, test_path


def load_data(y_name='Species'):
    """Returns the iris dataset as (train_x, train_y), (test_x, test_y)."""
    train_path, test_path = maybe_download()

    train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0, dtype=np.float64)
    train_x, train_y = train, train.pop(y_name)

    test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0, dtype=np.float64)
    test_x, test_y = test, test.pop(y_name)

    train_y = train_y.astype('int32')
    test_y = test_y.astype('int32')
    
    return (train_x, train_y), (test_x, test_y)


def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the dataset.
    return dataset


def eval_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction"""
    features=dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the dataset.
    return dataset


def read_mnist(batch_size, train=True):
    if train:
        filename_image_queue = mnist_train_filename_image_list
        filename_label_queue = mnist_train_filename_label_list
    else:
        filename_image_queue = mnist_eval_filename_image_list
        filename_label_queue = mnist_eval_filename_label_list

    class MnistRecord(object):
        pass

    result = MnistRecord()

    label_bytes = 1
    result.height = 28
    result.width = 28
    result.depth = 1

    image_bytes = result.height * result.width * result.depth

    image_dataset = tf.data.FixedLengthRecordDataset(filename_image_queue, record_bytes=image_bytes, header_bytes=16)
    label_dataset = tf.data.FixedLengthRecordDataset(filename_label_queue, record_bytes=label_bytes, header_bytes=8)

    image_dataset = image_dataset.map(lambda x: tf.decode_raw(x, tf.uint8)).map(lambda x: tf.cast(x, tf.float32))
    image_dataset = image_dataset.map(lambda x: tf.reshape(x, [result.depth, result.width, result.height]))
    image_dataset = image_dataset.map(lambda x: tf.transpose(x, [2, 1, 0]))
    
    label_dataset = label_dataset.map(lambda x: tf.decode_raw(x, tf.uint8)).map(lambda x: tf.cast(x, tf.int32))
    label_dataset = label_dataset.map(lambda x: tf.one_hot(x[0], depth=10))
    
    final_dataset = tf.data.Dataset.zip((image_dataset, label_dataset))

    if train:
        final_dataset = final_dataset.repeat().batch(batch_size)
    else:
        final_dataset = final_dataset.batch(batch_size)

    return final_dataset