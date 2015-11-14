__author__ = 'tiago'

from PIL import Image, ImageOps
import numpy as np
import os

class FaceContainer(object):
	def __init__(self, maxshift=0, cutoff=0, feature='gender'):
		self.feature=feature
		if feature=='gender':
			self.target_names=['male', 'female']
		elif feature=='smile':
			self.target_names=['neutral', 'smiling']
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

		
		# import faces:
		self.X = []
		self.y = [] # 0 is no smile, 1 is a smile
		# indices of the female pictures:
		self.indices_female=[ 10,	11,	14,	20,	25,	27,	28,	36,	38,	42,	50,	57,	59, 68,	71,	74,	77,	78,	83,	84,	85,	86,	90,	95,	96,	97, 98,	99, 100, 101, 102, 103, 104, 105, 116, 118, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 148, 149, 150, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 165, 166, 167, 168, 169, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 182, 183, 188, 194, 195, 196, 197, 198, 199]
		
		self.N=200 #number of faces used
		
		# translate faces by random distances 
		self.maxshift=maxshift # maximum random shift applied
		if cutoff < self.maxshift:
			cutoff=self.maxshift # number of pixels to cut of at each boarder, the used pixels will have an y-range of [cutoff+yshift; yshift-cutoff] with different yshift for each image
		self.cutoff=cutoff #not used here, but allows to access cutoffs of different face containers
		xshift=np.random.randint(-maxshift,maxshift+1, self.N) # different shifts for the images to challenge the classifier
		yshift=np.random.randint(-maxshift,maxshift+1, self.N)
		im = Image.open(path+str(1)+expression+'.jpg', 'r')
		self.imagesize=im.size
		for faceid in range(1, self.N+1):
			expression='a' #a is neutral, b is smiling
			im = Image.open(path+str(faceid)+expression+'.jpg', 'r')
			#pix=im.load()
			pix=np.array(im.getdata()) # Image has its own format, we want an array to do the normalization and shifts, note the array is linearized and not 2 dimensional!!!
			temp=(pix - np.mean(pix))*1.0/np.std(pix) #standardize mean and variance of pixel brightness
			temp=np.reshape(temp, [self.imagesize[0], self.imagesize[1]], order='F') # make the array 2d, but PIL linearizes along columns and not rows, two solutions: work with transposed data, or use the order='F' in reshape, 'F'=Fortran style arrays
			temp=temp[(cutoff+xshift[faceid-1]):(self.imagesize[0]-cutoff+xshift[faceid-1]), (cutoff+yshift[faceid-1]):(self.imagesize[1]-cutoff+yshift[faceid-1])]
			temp=np.reshape(temp, temp.size, order='F')
			self.X.append(temp)
			self.y.append(0)

		for faceid in range(1, self.N+1):
			expression='b' #a is neutral, b is smiling
			im = Image.open(path+str(faceid)+expression+'.jpg', 'r')
			#pix=im.load()
			pix=np.array(im.getdata())
			temp=(pix - np.mean(pix))*1.0/np.std(pix)
			temp=np.reshape(temp, [self.imagesize[0], self.imagesize[1]], order='F') # indeed PIL linearizes along columns and not rows, two solutions: work with transposed data, or use the order='F' in reshape, 'F'=Fortran style arrays
			temp=temp[(cutoff+xshift[faceid-1]):(self.imagesize[0]-cutoff+xshift[faceid-1]), (cutoff+yshift[faceid-1]):(self.imagesize[1]-cutoff+yshift[faceid-1])]
			temp=np.reshape(temp, temp.size, order='F')
			self.X.append(temp)
			self.y.append(1)
		if cutoff>0:
			self.imagesize=(self.imagesize[0]-2*cutoff, self.imagesize[1]-2*cutoff)
		# END translate faces by random distances 
	
	# function that returns data sorted for smile detection or gender detection
	def getTestTrainData(self):
		if self.feature=='smile':
			#train and test set for smiles:
			Xtrain=self.X[0::2]
			ytrain=self.y[0::2]
			Xtest=self.X[1::2]
			ytest=self.y[1::2]
		elif self.feature=='gender':
			#train and test set for gender:
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