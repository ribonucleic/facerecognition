__author__ = 'tiago'

import matplotlib.pyplot as plt
import argparse
from scipy.signal import gaussian
from theano.tensor.nnet import conv
import theano
import theano.tensor as T
import numpy as np
import os
import re
from collections import OrderedDict
from ioGenerator import ioGenerator
import warnings
import logging
from face_container import FaceContainer
from ann_utilities import *

def relu(x):
    return T.switch(x<0, 0, x)

np.random.seed(0)
rng = np.random.RandomState(1234)

fc = FaceContainer()
Xtrain, ytrain, Xtest, ytest = fc.getTestTrainData()

Xtrain = np.array(Xtrain).reshape(100, 1, fc.imagesize[0], fc.imagesize[1])
Xtest = np.array(Xtest).reshape(100, 1, fc.imagesize[0], fc.imagesize[1])
train_set_x = theano.shared(Xtrain.astype('float32'))
train_set_y = theano.shared(ytrain.astype('int32'))
test_set_x = theano.shared(Xtest.astype('float32'))
test_set_y = theano.shared(ytest.astype('int32'))

image_vector = T.ftensor4()
nr_filters = (4,1)
filter_size = (9,9)
pool_width = (2,2)
batch_size = 5
n_hidden = 50
target_vector = T.ivector()
index = T.iscalar()

layer1 = LeNetConvPoolLayer(
	rng,
	input=image_vector,
	filter_shape=(nr_filters[0], 1, filter_size[0], filter_size[0]),
	image_shape=(batch_size, 1, fc.imagesize[0], fc.imagesize[1]),
	poolsize=(pool_width[0],pool_width[0])
)

nwidth = (fc.imagesize[0]-filter_size[0]+1)/pool_width[0]
nheight = (fc.imagesize[1]-filter_size[0]+1)/pool_width[0]
layer2 = LeNetConvPoolLayer(
	rng,
	input=layer1.output,
	filter_shape=(nr_filters[1], nr_filters[0], filter_size[1], filter_size[1]),
	image_shape=(batch_size, nr_filters[0], nwidth, nheight),
	poolsize=(pool_width[1], pool_width[1])
)

nwidth = (nwidth-filter_size[1]+1)/pool_width[1]
nheight = (nheight-filter_size[1]+1)/pool_width[1]
layer3 = HiddenLayer(
	rng,
	input=layer2.output.flatten(2),
	n_in=nr_filters[1] * nwidth * nheight,
	n_out=n_hidden,
	activation=relu
)

layer4 = LogisticRegression(input=layer3.output, n_in=n_hidden, n_out=2)

cost = layer4.negative_log_likelihood(target_vector)

params = layer4.params + layer3.params + layer2.params + layer1.params

updates, _, _, _, _ = create_optimization_updates(cost, params, method='sgd')

train_model = theano.function(
	[index],
	cost,
	updates=updates,
	givens={
		image_vector: train_set_x[index * batch_size: (index + 1) * batch_size],
		target_vector: train_set_y[index * batch_size: (index + 1) * batch_size]
	}
)

test_model = theano.function(
	[index],
	layer4.errors(target_vector),
	givens={
		image_vector: test_set_x[index * batch_size: (index + 1) * batch_size],
		target_vector: test_set_y[index * batch_size: (index + 1) * batch_size]
	}
)

for i in xrange(20):
	print train_model(i)