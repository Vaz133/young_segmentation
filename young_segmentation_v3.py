from optparse import OptionParser
from skimage import io, color, filters
import numpy as np 
import itertools as IT
import matplotlib.pyplot as plt 
from PIL import Image
from matplotlib import image
from sklearn.cluster import KMeans, MiniBatchKMeans
from scipy.cluster import vq
import cv2
import time

start_time = time.time()

np.set_printoptions(threshold=np.nan)

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
		self.im_grey = cv2.imread(image,0)

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

	def visualize2(self):
		labels = S.kmeans().reshape(self.x, self.y)
		intensity_list = []
		final_list = []
		for i in range(3):
			im, _ = S.rgb2lab()
			im[labels != i] = 0
			final_list.append(im)
			intensity_list.append(np.mean(im))
		im_bin = final_list[intensity_list.index(min(intensity_list))]
		im_bin[im_bin != 0] = 1
		plt.imsave("./testing11.png", im_bin, cmap='gray')



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
		im[im != 0] = 1
		plt.imsave("./testing1_mb.png", im, cmap="gray")
		final[im == 0] = 0
		plt.imsave("./testing1_color.png", final)




(options, args) = parser.parse_args()

S = Segmentation(options.image, options.output_dir)
S.visualize()
print time.time() - start_time


