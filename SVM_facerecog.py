#!/usr/bin/env python

# this code trains a support vector machine (SVM) to recognize whether a face is male or female, or smiling or not, or some binary decision
# the SVM algorithm is the one from scikit-learn (sklearn module below)
# the faces are manually centered
# as a first experiment, we will offset the face positions by varying distances and see how soon the SVM fails
# we will compare this to a deep learning network, it should proof more resilient to offsets

from PIL import Image, ImageOps
from sklearn import svm
import numpy as np
from face_container import FaceContainer

np.random.seed(0)

fc = FaceContainer()
Xtrain, ytrain, Xtest, ytest = fc.getTestTrainData()

#train support vector machine:
clf = svm.SVC(kernel='linear') #classifier
print clf.fit(Xtrain, ytrain)

#test success:
yout=clf.predict(Xtest)

print sum(abs(np.array(yout)-np.array(ytest)))
print yout
print ytest

# now plot the separating face:
w = clf.coef_[0]
t=clf.intercept_[0]
img = Image.new( 'L', fc.imagesize, "black") # create a new black image
pixels = img.load() # create the pixel map
offs=min(w)
scale=255.0/max(w-offs)

for i in range(fc.imagesize[0]):    # for every pixel:
    for j in range(fc.imagesize[1]):
        pixels[i,j] = 255-scale*(w[fc.imagesize[0]*j +i]-offs) # set the colour accordingly

#img=ImageOps.equalize(img)
img.show()
