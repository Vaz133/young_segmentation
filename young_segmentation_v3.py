from optparse import OptionParser
from skimage import io, color, filters
import numpy as np 
import itertools as IT
import matplotlib.pyplot as plt 
from PIL import Image
from matplotlib import image
from sklearn.cluster import KMeans, MiniBatchKMeans
from scipy.cluster import vq
from scipy import ndimage
from skimage.measure import label, regionprops
import pandas as pd
from collections import defaultdict
import time

start_time = time.time()

# np.set_printoptions(threshold=np.nan)

parser = OptionParser()
parser.add_option("-i", "--input_image", dest="image", type="string", action="store")
parser.add_option("-o", "--output_directory", dest="output_dir", type="string", action="store")

class Segmentation:

	def __init__(self, image, output_dir):
		self.image = io.imread(image)
		self.output_dir = output_dir
		self.x = self.image.shape[0]
		self.y = self.image.shape[1]
		self.dimensions = self.image.shape[0] * self.image.shape[1]
		self.im_grey = ndimage.imread(image)[:,:,0]

	def rgb2lab(self):
		lab = color.rgb2lab(self.image)
		nuclei = lab[:,:,2]
		lab1 = np.reshape(lab[:,:,0], (1, self.dimensions))
		lab2 = np.reshape(lab[:,:,1], (1, self.dimensions))
		lab3 = np.reshape(lab[:,:,2], (1, self.dimensions))
		lab_flattened = np.vstack((vq.whiten(lab1),vq.whiten(lab2),vq.whiten(lab3)))
		return nuclei, lab_flattened

	def extract_gabor(self):
		im_grey, lab_features = S.rgb2lab()
		frequencies = [1.571]
		orientation_rads = [0, 0.785398, 1.5708, 2.3562]
		combos = list(IT.product(frequencies, orientation_rads))
		list_of_mags = []
		for item in combos:
			real_gabor, imaginary_gabor = filters.gabor(im_grey, item[0], item[1])
			real_gabor_flattened = np.reshape(real_gabor, (1, (real_gabor.shape[0]*real_gabor.shape[1])))
			imaginary_gabor_flattened = np.reshape(imaginary_gabor, (1, (imaginary_gabor.shape[0]*imaginary_gabor.shape[1])))
			mag = np.sqrt(np.square(real_gabor_flattened, dtype=np.float64)+np.square(imaginary_gabor_flattened, dtype=np.float64))
			sigma = 2 * item[0]
			K = 2
			mag_gaussian = vq.whiten(filters.gaussian(mag, K*sigma))
			list_of_mags.append(mag_gaussian)
		list_of_mags = np.asarray(list_of_mags).reshape(len(combos),self.dimensions)
		features = np.vstack((lab_features, list_of_mags))
		return features

	def kmeans(self):
		features = S.extract_gabor().T
		mkb = MiniBatchKMeans(n_clusters=3)
		mkb.fit(features)
		return mkb.labels_

	def visualize(self):
		labels = S.kmeans()
		labels = labels.reshape(self.x, self.y)
		intensity_list = []
		final_list = []
		final = self.image
		for i in range(3):
			grey = self.im_grey.copy()
			grey[labels != i] = 0
			final_list.append(grey)
			intensity_list.append(np.mean(grey))
		min_index = intensity_list.index(min(intensity_list))
		im = final_list[min_index]
		im_grey = np.zeros((self.x, self.y))
		im_grey[im != 0] = 1
		plt.imsave("./testing1_mb.png", im_grey, cmap="gray")
		final[im == 0] = 0
		plt.imsave("./testing1_color.png", final)
		return im_grey

	def labels(self):
		img = S.visualize()
		labels = label(img)
		measurements = regionprops(labels, self.im_grey)
		print len(measurements)
		meas_dict = defaultdict(list)
		for i in range(len(measurements)):
			meas_dict['area'].append(measurements[i]['area'])
			meas_dict['bbox'].append(measurements[i]['bbox'])
			meas_dict['centroid'].append(measurements[i]['centroid'])
			meas_dict['convex_area'].append(measurements[i]['convex_area'])
			meas_dict['convex_image'].append(measurements[i]['convex_image'])
			meas_dict['coords'].append(measurements[i]['coords'])
			meas_dict['eccentricity'].append(measurements[i]['eccentricity'])
			meas_dict['equivalent_diamterer'].append(measurements[i]['equivalent_diameter'])
			meas_dict['euler_number'].append(measurements[i]['euler_number'])
			meas_dict['extent'].append(measurements[i]['extent'])
			meas_dict['filled_area'].append(measurements[i]['filled_area'])
			meas_dict['filled_image'].append(measurements[i]['filled_image'])
			meas_dict['image'].append(measurements[i]['image'])
			meas_dict['intertia_tensor'].append(measurements[i]['inertia_tensor'])
			meas_dict['intertia_tensor_eigvals'].append(measurements[i]['inertia_tensor_eigvals'])
			meas_dict['intensity_image'].append(measurements[i]['intensity_image'])
			meas_dict['label'].append(measurements[i]['label'])
			meas_dict['local_centroid'].append(measurements[i]['local_centroid'])
			meas_dict['major_axis_length'].append(measurements[i]['major_axis_length'])
			meas_dict['max_intensity'].append(measurements[i]['max_intensity'])
			meas_dict['mean_intensity'].append(measurements[i]['mean_intensity'])
			meas_dict['min_intensity'].append(measurements[i]['min_intensity'])
			meas_dict['minor_axis_length'].append(measurements[i]['minor_axis_length'])
			meas_dict['moments'].append(measurements[i]['moments'])
			meas_dict['moments_central'].append(measurements[i]['moments_central'])
			meas_dict['moments_hu'].append(measurements[i]['moments_hu'])
			meas_dict['moments_normalized'].append(measurements[i]['moments_normalized'])
			meas_dict['orientation'].append(measurements[i]['orientation'])
			meas_dict['perimeter'].append(measurements[i]['perimeter'])
			meas_dict['solidity'].append(measurements[i]['solidity'])
			meas_dict['weighted_centroid'].append(measurements[i]['weighted_centroid'])
			meas_dict['weighted_local_centroid'].append(measurements[i]['weighted_local_centroid'])
			meas_dict['weighted_moments'].append(measurements[i]['weighted_moments'])
			meas_dict['weighted_moments_central'].append(measurements[i]['weighted_moments_central'])
			meas_dict['weighted_moments_hu'].append(measurements[i]['weighted_moments_hu'])
			meas_dict['weighted_moments_normalized'].append(measurements[i]['weighted_moments_normalized'])
		meas_df = pd.DataFrame(meas_dict)
		# meas_df.to_csv("./test.csv")

(options, args) = parser.parse_args()

S = Segmentation(options.image, options.output_dir)
# S.visualize()
S.labels()
print time.time() - start_time


