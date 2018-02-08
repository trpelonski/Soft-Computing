import numpy as np
import cv2
import os, glob
import keras
import h5py
from keras.models import Sequential
from keras.layers.core import Dense,Activation, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, Adam


def readAllFiles(path):
	paths = []
	for infile in glob.glob(os.path.join(path, '*.jpg')):
		paths.append(infile)
	return paths

def resizeForNetwork(img):
	img_network = cv2.resize(img, (28, 28), interpolation = cv2.INTER_CUBIC)
	return img_network
	 
def findEdges(img):
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
	img_close = cv2.dilate(img, kernel, iterations=1) - cv2.erode(img, kernel, iterations=1)
	return img_close
	
def findKMeansFlower(img):
	Z = img.reshape((-1,3))
	Z = np.float32(Z)
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
	K = 2
	ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
	center = np.uint8(center)
	res = center[label.flatten()]
	res2 = res.reshape((img.shape))
	return res2
	
def findFlowerContour(img):
	img_cont, contours, hierarchy = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	sizes = []
	for contour in contours:
		center, size, angle = cv2.minAreaRect(contour)
		sizes.append(size)
    
	biggest_contour_index = sizes.index(max(sizes))
	biggest_contour = contours[biggest_contour_index]
	
	return biggest_contour

def readOutputsTest(path):
	outputs = []
	with open(path) as fp:
		line = fp.readline()
		while line:
			outputs.append(line)
			line = fp.readline()
	return outputs

def convertOutput(x):
	x = int(x)
	output = np.zeros(80)
	output[x-1] = 1
	
	return output
	
def outputToInt(output):
	
	converted = max(enumerate(output), key=lambda x: x[1])[0]
	
	return converted+1
	
	
def calculateError(x_real, x_res):
	retVal = abs(x_real - x_res)/x_real
	
	return (1 - retVal)*100
	
def calculateAccuracy(results):
	accuracy = 0
	for result in results:
		accuracy = accuracy + result
	accuracy = accuracy/len(results)
	return accuracy

	
def defineSegment(path):
	print(path)
	img = cv2.imread(path)
	res = findKMeansFlower(img)

	res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

	img_close = findEdges(res)

	x,y,w,h = cv2.boundingRect(findFlowerContour(img_close)) 
	region = img[y:y+h,x:x+w]    
	region = resizeForNetwork(region)
	
	return region		

def scale_to_range(image):
    return image/255

def matrix_to_vector(image):
    return image.flatten()

def prepare_for_ann(regions):
	ready_for_ann = []
	for region in regions:
		region = region.reshape(28,28,3)
		scale = scale_to_range(region)
		ready_for_ann.append(scale)
        
	return ready_for_ann

def createInputs(path):
	trainResault = readAllFiles(path)
	regions = []
	for path in trainResault:
		region = defineSegment(path)
		regions.append(region)

	trainInputs = prepare_for_ann(regions)
	return trainInputs

def createOutputs(path):
	trainOutputs = []
	outputsInt = readOutputsTest(path)
	for outputInt in outputsInt:
		trainOutputs.append(convertOutput(outputInt))
		
	return trainOutputs
	
def create_ann():
	
	ann = Sequential()
	
	ann.add(Convolution2D(32,3,data_format='channels_last',activation='relu',input_shape=(28,28,3)))
	ann.add(MaxPooling2D(pool_size=(2,2)))
	ann.add(Convolution2D(64,3, activation="relu"))
	ann.add(MaxPooling2D(pool_size=(2,2)))

	ann.add(Flatten())
	ann.add(Dense(128))
	ann.add(Dropout(0.5))
	ann.add(Dense(80))
	ann.add(Activation('softmax'))
	
	return ann
    
def train_ann(ann, X_train, y_train):
	X_train = np.array(X_train, np.float32)
	y_train = np.array(y_train, np.float32)
   
	adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
	ann.compile(loss='categorical_crossentropy', optimizer = adam, metrics = ['accuracy'])
	
	ann.fit(X_train, y_train, epochs=100, batch_size=46, verbose = 2, shuffle=False) 
      
	return ann


inputsForTest = createInputs('test')
outputsForTest = readOutputsTest("outputs_test.txt")

ann = create_ann()

fName = 'mreza_nova1.hdf5'

if os.path.exists(fName):
	ann.load_weights(fName)
else:
	inputsForTrain = createInputs('train')
	outputsForTrain = createOutputs("outputs_train.txt")
	ann = train_ann(ann, inputsForTrain, outputsForTrain)
	ann.save_weights(fName,overwrite=True)
	
results = ann.predict(np.array(inputsForTest, np.float32))
resultsPercentage = []

for index in range(len(results)):
	result = outputToInt(results[index])
	resultsPercentage.append(calculateError(int(outputsForTest[index]),result))
	print('Cvet broj: '+str(index+1)+ ' ima '+str(result)+' latica, treba da ima '+str(outputsForTest[index])+' Tacnost je '+str(resultsPercentage[index])+' %')

print('Ukupna tacnost je '+str(calculateAccuracy(resultsPercentage))+' %')

cv2.waitKey(0)
