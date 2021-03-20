
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
x = pickle.load(pickle_in)

for i in range(len(x)):
	x[i] = cv2.resize(x[i], (IMAGESIZEX,IMAGESIZEY))

x = np.array(x)
x = np.array(x).reshape(-1, IMAGESIZEY, IMAGESIZEX, 3)

x = x/255.0

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


#NN BUILDING
# here is some example of layer and node optimization https://pythonprogramming.net/using-trained-model-deep-learning-python-tensorflow-keras/
model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=x.shape[1:])) #X.shape[1,] => (140,100,3)
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(5000)) #10 000
model.add(Dense(7500)) #15 000
model.add(Dense(10000)) #20 000
#output layer
model.add(Dense(labels.shape[1]))
model.add(Activation('sigmoid')) # change to relu?

#compile the model
model.compile(loss='binary_crossentropy',
              optimizer='adamax',
              metrics=['binary_accuracy'])

#train the model
model.fit(x, labels, batch_size=8, epochs=20, validation_split=0.3,verbose=2,shuffle=True)

model.evaluate(x, labels, batch_size=10, verbose=1)

prediction = model.predict(x)
for i in range(len(prediction)):
	for j in range(len(prediction[i])):
		if prediction[i][j] > 0.5:
			prediction[i][j] = 1
		else:
			prediction[i][j] = 0

for i in range(len(prediction)):
	print(labels[i])
	print(prediction[i])

model.save('NN.model')









'''
output = model.predict(X)

for i in range(len(output)):
	for j in range(len(output[i])):
		if output[i][j] > 0.5:
			output[i][j] = 1
		else:
			output[i][j] = 0

for i in range(len(output)):
	print(labels[i])
	print(output[i])

output = model.predict(z)

for i in range(len(output)):
	for j in range(len(output[i])):
		if output[i][j] > 0.3:
			output[i][j] = 1
		else:
			output[i][j] = 0


for i in range(len(output)):
	print(labels[i])
	print(output[i])

model.evaluate(z, labels, batch_size=10, verbose=1)
'''










###################################################################################################

#optimization time!
'''
for i in [128,256,512]:
	for j in [128,256,512]:
		for k in [128,256,512]:
			for l in [128,256,512]:
				NAME = "{}-1st conv layer filters-{}-2nd conv layer filters-{}-1st dense layer nodes-{}-2nd dense layer nodes".format(i,j,k,l)
				print(NAME)
				model = Sequential()

				model.add(Conv2D(512, (3, 3), input_shape=X.shape[1:])) #X.shape[1,] => (140,100,3)
				model.add(Activation('relu'))
				model.add(MaxPooling2D(pool_size=(5, 5)))
				
				model.add(Conv2D(1024, (3, 3)))
				model.add(Activation('relu'))
				model.add(MaxPooling2D(pool_size=(2, 2)))

				model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

				model.add(Dense(512))
				model.add(Dense(1024))
				#output layer
				model.add(Dense(2))
				model.add(Activation('sigmoid'))
				tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))


				#compile the model
				model.compile(loss='binary_crossentropy',
				              optimizer='adam',
				              metrics=['accuracy'])
				
				#train the model
				model.fit(X, labels, batch_size=8, epochs=15, validation_split=0.3,callbacks=[tensorboard])

'''