import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import h5py
import numpy as np
import os
import glob
import cv2
from matplotlib import pyplot
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.externals import joblib


fixed_size = tuple((500, 500))


clf  = RandomForestClassifier(n_estimators=100,random_state=9)



h5f_data = h5py.File('output/data.h5', 'r')
h5f_label = h5py.File('output/labels.h5', 'r')

global_features_string = h5f_data['dataset_1']
global_labels_string = h5f_label['dataset_1']

global_features = np.array(global_features_string)
global_labels = np.array(global_labels_string)

h5f_data.close()
h5f_label.close()




clf.fit(global_features,global_labels)



train_path = "dataset/train"


train_labels = os.listdir(train_path)


train_labels.sort()



test_path = "dataset/test"


bins = 8





def fd_histogram(image, mask=None):
	image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	hist  = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
	hist = cv2.normalize(hist, hist)
	return hist.flatten()



for file in glob.glob(test_path + "/*.jpg"):
	image = cv2.imread(file)
	image = cv2.resize(image, fixed_size)
	fv_histogram  = fd_histogram(image)

	
	global_feature = np.hstack([fv_histogram])

	
	prediction = clf.predict(global_feature.reshape(1,-1))[0]

	
	cv2.putText(image, train_labels[prediction], (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 3)
    
	
	#plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
	#plt.show()
	#img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	cv2.imshow('Image',image)
	cv2.waitKey(1500)
	
