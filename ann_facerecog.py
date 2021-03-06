#!/usr/bin/python
#__author__ = 'tiago'

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
#from ioGenerator import ioGenerator
import warnings
import logging
from face_container import FaceContainer
from ann_utilities import *
#from theano.compile.debugmode import DebugMode

theano.config.floatX='float32'

def relu(x):
    return T.switch(x<0, 0, x)

np.random.seed(0)
rng = np.random.RandomState(1234)

fc = FaceContainer()
Xtrain, ytrain, Xtest, ytest = fc.getTestTrainData()
#ytrain = np.vstack((ytrain,1 - ytrain))  #to give two targets for the two softmax neurons
#ytest  = np.vstack((ytest, 1 - ytest))

Xtrain = np.array(Xtrain).reshape(100, 1, fc.imagesize[0], fc.imagesize[1])
Xtest = np.array(Xtest).reshape(100, 1, fc.imagesize[0], fc.imagesize[1])
train_set_x = theano.shared(Xtrain.astype('float32'))
train_set_y = theano.shared(ytrain.astype('int32'))
test_set_x = theano.shared(Xtest.astype('float32'))
test_set_y = theano.shared(ytest.astype('int32'))

#print type(ytrain)
#print np.shape(ytrain)
#print ytrain
#print train_set_y.shape.eval()
#exit()

image_vector = T.ftensor4() # 4 dim. array of floats, 
nr_filters = (4,1)
filter_size = (9,9)
pool_width = (2,2)
batch_size = 5
n_hidden = 50
target_vector = T.ivector() # vector of integers, fvector would be floats
index = T.iscalar()

layer1 = LeNetConvPoolLayer(
	rng,
	input=image_vector,
	filter_shape=(nr_filters[0], 1, filter_size[0], filter_size[0]),
	image_shape=(batch_size, 1, fc.imagesize[0], fc.imagesize[1]),
	poolsize=(pool_width[0], pool_width[0])
)

nwidth = (fc.imagesize[0]-filter_size[0]+1)/pool_width[0]
nheight = (fc.imagesize[1]-filter_size[0]+1)/pool_width[0] #ERROR? filter_size[0]
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

layer4 = LogisticRegression(rng, input=layer3.output, n_in=n_hidden, n_out=2)


params = layer4.params + layer3.params + layer2.params + layer1.params

#L2_sqr = ((layer1.W ** 2).sum() + (layer2.W ** 2).sum() + (layer3.W ** 2).sum() + (layer4.W ** 2).sum())

cost = layer4.negative_log_likelihood(target_vector) #+ 10*L2_sqr


updates, _, _, _, _ = create_optimization_updates(cost, params, method='sgd', lr=0.001, eps= 1e-5)


train_model = theano.function(
	[index],
	cost,
	updates=updates,
	givens={
		image_vector: train_set_x[index * batch_size: (index + 1) * batch_size],
		target_vector: train_set_y[index * batch_size: (index + 1) * batch_size]
	}
	#mode='DebugMode'
)

test_model = theano.function(
	[index],
	layer4.errors(target_vector),
	givens={
		image_vector: test_set_x[index * batch_size: (index + 1) * batch_size],
		target_vector: test_set_y[index * batch_size: (index + 1) * batch_size]
	}
)
	




for i in range(20):
	print train_model(i)
	#print test_model(i)
	
for i in range(20):
	print train_model(i)
for i in range(20):
	print train_model(i)
for i in range(20):
	print train_model(i)
for i in range(20):
	print train_model(i)
for i in range(20):
	print train_model(i)
for i in range(20):
	print train_model(i)
for i in range(20):
	print train_model(i)

for i in range(20):
	print test_model(i)
