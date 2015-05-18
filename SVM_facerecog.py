#!/usr/bin/env python

# this code trains a support vector machine (SVM) to recognize whether a face is male or female, or smiling or not, or some binary decision
# the SVM algorithm is the one from scikit-learn (sklearn module below)
# the faces are manually centered
# as a first experiment, we will offset the face positions by varying distances and see how soon the SVM fails
# we will compare this to a deep learning network, it should proof more resilient to offsets

from PIL import Image, ImageOps
from sklearn import svm
import numpy as np
import os

np.random.seed(0)
path='faces/bw/'
expression='a' #a is neutral, b is smiling
faceid='1' #integer from 1 to 200, there are 200 faces in "frontalimages_manuallyaligned"

if not os.path.isdir('faces/bw'):
  os.mkdir('faces/bw')
  for myfile in os.listdir('faces/color'):
    print myfile
    img = Image.open('faces/color/'+myfile).convert('L')
    img.save('faces/bw/'+myfile)

#print pix[5,10] #Get the RGBA Value of the a pixel of an image, or the brightness for the folders that are black and white (they have a '_bw' suffix)
# to transform RGB to black/white use: sqrt(0.299 * R^2 + 0.587 * G^2 + 0.114 * B^2)
# the formula is from this discussion: http://stackoverflow.com/questions/596216/formula-to-determine-brightness-of-rgb-color
# to save time, I created black and white versions of the FEI_Face_database, see above at path=...
# the bash command to make the pictures black and white in the shell was this:
# for N in {1..200}; do convert frontalimages_manuallyaligned/${N}a.jpg -colorspace Gray frontalimages_manuallyaligned_bw/${N}a.jpg;  done
# see also bash example folder for_loop_basic.sh

# import faces:
X = []
y=[] # 0 is no smile, 1 is a smile
# indices of the female pictures:
indices_female=[ 10,  11,  14,  20,  25,  27,  28,  36,  38,  42,  50,  57,  59, 68,  71,  74,  77,  78,  83,  84,  85,  86,  90,  95,  96,  97, 98,  99, 100, 101, 102, 103, 104, 105, 116, 118, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 148, 149, 150, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 165, 166, 167, 168, 169, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 182, 183, 188, 194, 195, 196, 197, 198, 199]

N=200 #number of faces used
for faceid in range(1, N+1):
  expression='a' #a is neutral, b is smiling
  im = Image.open(path+str(faceid)+expression+'.jpg', 'r')
  #pix=im.load()
  pix=np.array(im.getdata())
  temp=(pix - np.mean(pix))*1.0/np.std(pix)
  X.append(temp)
  y.append(0)
  
for faceid in range(1, N+1):
  expression='b' #a is neutral, b is smiling
  im = Image.open(path+str(faceid)+expression+'.jpg', 'r')
  #pix=im.load()
  pix=np.array(im.getdata())
  temp=(pix - np.mean(pix))*1.0/np.std(pix)
  X.append(temp)
  y.append(1)

# train and test set for smiles:
#Xtrain=X[0::2]
#ytrain=y[0::2]
#Xtest=X[1::2]
#ytest=y[1::2]
# train and test set for gender:
Xgender=X[0:N]
ygender=[0]*N # 0 is male
ygender=np.array(ygender)
ygender[indices_female]=1
permutation=list(np.random.permutation(N))
Xgender=[Xgender[i] for i in permutation]
ygender=ygender[permutation]
Xtrain=Xgender[0:N/2]
ytrain=ygender[0:N/2]
Xtest=Xgender[N/2:]
ytest=ygender[N/2:]

#train support vector machine:
clf = svm.SVC(kernel='linear') #classifier
print clf.fit(Xtrain, ytrain)

#test success:
yout=clf.predict(Xtest)

print sum(abs(np.array(yout)-np.array(ytest)))
print yout
print ytest

# now plot the separating face:
im = Image.open(path+str(1)+expression+'.jpg', 'r')
imagesize=im.size
w = clf.coef_[0]
t=clf.intercept_[0]
img = Image.new( 'L', imagesize, "black") # create a new black image
pixels = img.load() # create the pixel map
offs=min(w)
scale=255.0/max(w-offs)

for i in range(imagesize[0]):    # for every pixel:
    for j in range(imagesize[1]):
        pixels[i,j] = 255-scale*(w[imagesize[0]*j +i]-offs) # set the colour accordingly

#img=ImageOps.equalize(img)
img.show()
