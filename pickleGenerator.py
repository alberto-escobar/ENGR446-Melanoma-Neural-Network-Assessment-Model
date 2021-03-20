#this is the main code for the ENGR 446 melanoma detection using NN project.
#Written by Alberto Escobar
#
#Reference material to help code
#https://pythonprogramming.net/loading-custom-data-deep-learning-python-tensorflow-keras/?completed=/introduction-deep-learning-python-tensorflow-keras/

#imports
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle

#GLOBAL VARIABLES
#700 and 500
#or 140 and 100
#IMAGESIZEX = 140
#IMAGESIZEY = 100
LABELS_FILE_PATH = "PH2Dataset\PH2_dataset3.csv"

#LABELS GENERATION
#Take labels data from csv file and input into array, here the size of the 
labels = np.genfromtxt(LABELS_FILE_PATH,delimiter=',',dtype=None)

for i in range(len(labels)):
	for j in range(len(labels[i])):
		labels[i][j] = labels[i][j].decode('UTF-8')

#pickle headers
pickle_out = open("labels_headers.pickle","wb")
pickle.dump(labels[0], pickle_out)
pickle_out.close()

#pickle labels
labels = np.delete(labels, 0, axis=0)
pickle_out = open("labels.pickle","wb")
pickle.dump(labels, pickle_out)
pickle_out.close()
#end of LABELS GENERATION

#FEATURES GENERATION
'''
#parse through dataset for images of moles, store the features into an array
training_data = []
features = []
testing = []
i = 1
for subdir, dirs, files in os.walk(os.getcwd()):
	for filename in files:
		filepath = subdir + os.sep + filename
		if filepath.endswith(".bmp") and len(filepath) == 91:
			print(filepath)
			print(labels[i][0])
			img_array = cv2.imread(filepath)
			new_array = cv2.resize(img_array, (IMAGESIZEX,IMAGESIZEY))
			flipped_array = cv2.flip(new_array, 0)
			flipped_again_array = cv2.flip(flipped_array, 1)
			features.append(new_array)
			testing.append(flipped_array)
			training_data.append([new_array,flipped_array,labels[i]])
			i = i+1
		else:
			pass

cv2.imshow('image1',new_array)
cv2.imshow('image2',flipped_again_array)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''

#parse through dataset for images of moles, store the images as arrays into an array
features = []
i = 0
for subdir, dirs, files in os.walk(os.getcwd()):
	for filename in files:
		filepath = subdir + os.sep + filename
		if filepath.endswith(".bmp") and len(filepath) == 91:
			print(filepath)
			print(labels[i][0])
			image = cv2.imread(filepath)
			features.append(image)
			i = i+1
		else:
			pass

#pickle the array of images 
pickle_out = open("features.pickle","wb")
pickle.dump(features, pickle_out)
pickle_out.close()
#end of FEATURES GENERATION

'''

#SHUFFLE DATA
#shuffle our data
random.shuffle(training_data)
#end of SHUFFLE DATA

#PICKLE GENERATION
#seperate shuffled data
X = []
y = []
z = []

for j in range(len(training_data)):
	X.append(training_data[j][0])
	y.append(training_data[j][2])
	z.append(training_data[j][1])

#printed before and after, this seems to be useless
X = np.array(X).reshape(-1, IMAGESIZEX, IMAGESIZEY, 3)
z = np.array(z).reshape(-1, IMAGESIZEX, IMAGESIZEY, 3)

#create pickles of the seperated data
pickle_out = open("X.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()

pickle_out = open("z.pickle","wb")
pickle.dump(z, pickle_out)
pickle_out.close()
#end of PICKLE GENERATION

#DEBUG
print(labels[0])
#A quick print out of the data length 
print(len(training_data))
print(len(X))
print(len(y))
print(len(z))
#end of DEBUG
'''