__author__ = 'tiago'

from PIL import Image, ImageOps
import numpy as np
import os

class FaceContainer(object):
	def __init__(self):
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
		# for N in {1..200}; do convert frontalimages_manuallyaligned/${N}a.jpg -colorspace Gray frontalimages_manuallyaligned_bw/${N}a.jpg;	done
		# see also bash example folder for_loop_basic.sh

		# import faces:
		self.X = []
		self.y = [] # 0 is no smile, 1 is a smile
		# indices of the female pictures:
		self.indices_female=[ 10,	11,	14,	20,	25,	27,	28,	36,	38,	42,	50,	57,	59, 68,	71,	74,	77,	78,	83,	84,	85,	86,	90,	95,	96,	97, 98,	99, 100, 101, 102, 103, 104, 105, 116, 118, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 148, 149, 150, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 165, 166, 167, 168, 169, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 182, 183, 188, 194, 195, 196, 197, 198, 199]

		self.N=200 #number of faces used
		im = Image.open(path+str(1)+expression+'.jpg', 'r')
		self.imagesize=im.size
		for faceid in range(1, self.N+1):
			expression='a' #a is neutral, b is smiling
			im = Image.open(path+str(faceid)+expression+'.jpg', 'r')
			#pix=im.load()
			pix=np.array(im.getdata())
			temp=(pix - np.mean(pix))*1.0/np.std(pix)
			self.X.append(temp)
			self.y.append(0)

		for faceid in range(1, self.N+1):
			expression='b' #a is neutral, b is smiling
			im = Image.open(path+str(faceid)+expression+'.jpg', 'r')
			#pix=im.load()
			pix=np.array(im.getdata())
			temp=(pix - np.mean(pix))*1.0/np.std(pix)
			self.X.append(temp)
			self.y.append(1)

	def getTestTrainData(self):
		# train and test set for smiles:
		#Xtrain=X[0::2]
		#ytrain=y[0::2]
		#Xtest=X[1::2]
		#ytest=y[1::2]
		# train and test set for gender:
		Xgender=self.X[0:self.N]
		ygender=[0]*self.N # 0 is male
		ygender=np.array(ygender)
		ygender[self.indices_female]=1
		permutation=list(np.random.permutation(self.N))
		Xgender=[Xgender[i] for i in permutation]
		ygender=ygender[permutation]
		Xtrain=Xgender[0:self.N/2]
		ytrain=ygender[0:self.N/2]
		Xtest=Xgender[self.N/2:]
		ytest=ygender[self.N/2:]
		return Xtrain, ytrain, Xtest, ytest