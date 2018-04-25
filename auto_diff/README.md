## About
The files in this folder are for Convolutional Neural Network written using Automatic Differentiation of Tensorflow

The entire code is written using **Python 3.6**

You can see all the available options using command `python train.py -h`

To evaluate a model, you can see all the options using `python eval.py -h`

To train the model using caltech_101 dataset, you will need to place the datasets in a *datasets/caltech_101* folder in the parent directory. Similarly for mnist dataset in *datasets/mnist* in parent directory.

## How to run

To start training on the mnist dataset, once dataset is set, enter following command

`python train.py caltech_101`

The above command will train single stage network using average pooling and complete supervised training.

To add a Rectified Linear Unit layer, add --relu option.
To add a Local Response Normalization layer, add --lrn option.
To perform supervised training only on the last layer, add --no-train option.

Once training is done, to perform evaluation, enter following command

`python eval.py caltech_101 -batch_size 102`

The evaluation prints precision score in the first row, recall score in the second row and accuracy score in the third row.