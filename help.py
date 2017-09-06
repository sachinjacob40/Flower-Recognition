
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib
from sklearn import datasets
from skimage.feature import hog
import numpy as np
import mahotas
import cv2
import os
import h5py


fixed_size = tuple((500, 500))


train_path = "dataset/train"


bins = 8


def fd_histogram(image, mask=None):
	
	image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	
	hist  = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
	
	hist = cv2.normalize(hist, hist)
	
	return hist.flatten()



train_labels = os.listdir(train_path)


train_labels.sort()
print(train_labels)


global_features = []
labels = []

i, j = 0, 0
k = 0



for training_name in train_labels:
	
	dir = os.path.join(train_path, training_name)

	current_label = training_name

	k = 1
	
	for x in range(1,201):
		
		file = dir + "/" + str(x) + ".jpg"
		
		image = cv2.imread(file)
		
		image = cv2.resize(image, fixed_size)
		
		fv_histogram = fd_histogram(image)

		
		global_feature = np.hstack([fv_histogram])

		labels.append(current_label)
		global_features.append(global_feature)
		i += 1
		k += 1

	print "[STATUS] processed folder: {}".format(current_label)
	j += 1

print "[STATUS] completed Feature Extraction..."



print "[STATUS] feature vector size {}".format(np.array(global_features).shape)


print "[STATUS] training Labels {}".format(np.array(labels).shape)



targetNames = np.unique(labels)
le = LabelEncoder()
target = le.fit_transform(labels)
print "[STATUS] training labels encoded..."



scaler = MinMaxScaler(feature_range=(0, 1))
rescaled_features = scaler.fit_transform(global_features)
print "[STATUS] feature vector normalized..."

print "[STATUS] target labels: {}".format(target)
print "[STATUS] target labels shape: {}".format(target.shape)



h5f_data = h5py.File('output/data.h5', 'w')
h5f_data.create_dataset('dataset_1', data=np.array(rescaled_features))

h5f_label = h5py.File('output/labels.h5', 'w')
h5f_label.create_dataset('dataset_1', data=np.array(target))

h5f_data.close()
h5f_label.close()

print "[STATUS] end of training.."


