import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import pickle
import numpy as np
import cv2
import random

print("hello world, this code can run!")
IMAGESIZEX = 140
IMAGESIZEY = 100

'''
#load the pickels
pickle_in = open("X.pickle","rb")
X = pickle.load(pickle_in)

#normalize the features
X = X/255.0

pickle_in = open("z.pickle","rb")
z = pickle.load(pickle_in)
z = z/255.0


pickle_in = open("y.pickle","rb")
y = pickle.load(pickle_in)
'''

#IMAGE PREPROCESSING
pickle_in = open("features.pickle","rb")
features = pickle.load(pickle_in)

z = []
for i in range(len(features)):
	image = cv2.flip(features[i], 0)
	image = cv2.flip(image, 1)
	z.append(cv2.resize(image, (IMAGESIZEX,IMAGESIZEY)))
z = np.array(z)
z = np.array(z).reshape(-1, IMAGESIZEY, IMAGESIZEX, 3)

z = z/255.0

#LABELS PRECPROCESSING
#so the output should be asym, pigment,streaks, regression, viel
pickle_in = open("labels.pickle","rb")
y = pickle.load(pickle_in)

labels = []

for i in range(len(y)):
	labels.append(
		[y[i][2].decode('UTF-8'),	#Asymmetry
		y[i][3].decode('UTF-8'),	#Pigment Network
		y[i][4].decode('UTF-8'),	#Regression Areas
		y[i][5].decode('UTF-8'),	#Blue-Whitish Veil
		y[i][6].decode('UTF-8'),	#White
		y[i][7].decode('UTF-8'),	#Red
		y[i][8].decode('UTF-8'),	#Light-Brown
		y[i][9].decode('UTF-8'),	#Dark-Brown
		y[i][10].decode('UTF-8'),	#Blue-Gray
		y[i][11].decode('UTF-8')])	#Black

for i in range(len(labels)):
	for j in range(len(labels[i])):
		if (labels[i][j] == '1'):
			labels[i][j] = 1
		if (labels[i][j] == '0'):
			labels[i][j] = 0	

labels = np.array(labels)


shuffle = []
for i in range(len(z)):
	shuffle.append([features[i],z[i],labels[i]])

random.shuffle(shuffle)


features = []
z = []
labels = []

for j in range(len(shuffle)):
	features.append(shuffle[j][0])
	z.append(shuffle[j][1])
	labels.append(shuffle[j][2])

z=np.array(z)
labels = np.array(labels)

model = tf.keras.models.load_model("NN.model")

model.evaluate(z, labels, batch_size=10, verbose=1)

prediction = model.predict(z)
for i in range(len(prediction)):
	for j in range(len(prediction[i])):
		if prediction[i][j] > 0.5:
			prediction[i][j] = 1
		else:
			prediction[i][j] = 0

for i in range(len(prediction)):
	print(labels[i])
	print(prediction[i])


#remember that features is not flipped like z is
while True:
	j=random.randint(0,199)
	print("###############################")

	string = ["Asymmetry ",
		"Pigment Network ",
		"Regression Areas ",
		"Blue-Whitish Veil ",
		"White ",
		"Red ",
		"Light-Brown ",
		"Dark-Brown ",
		"Blue-Gray ",
		"Black "]
	for i in range(len(string)):
		image = cv2.putText(features[j], str(prediction[j][i])+" "+string[i], (50,(50+(i*15))), cv2.FONT_HERSHEY_SIMPLEX , 0.4, (255,0,0), 1, cv2.LINE_AA) 
	cv2.imshow('image',image)
	cv2.imshow('input',z[j])
	print(j)
	print(string)
	print(labels[j])
	print(prediction[j])

	cv2.waitKey(0)
	cv2.destroyAllWindows()


