from optparse import OptionParser
from skimage import io, color, filters
import numpy as np 
import itertools as IT
import matplotlib.pyplot as plt 
from PIL import Image
from matplotlib import image
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import scale, normalize
from scipy.cluster import vq
from scipy import ndimage, misc
from skimage.measure import label, regionprops
import pandas as pd
from collections import defaultdict
import time
import cPickle as pickle
from math import pi
from skimage.feature import peak_local_max, structure_tensor, structure_tensor_eigvals
from skimage.morphology import watershed, disk, white_tophat, black_tophat, remove_small_objects
from skimage.morphology import erosion, dilation, opening, skeletonize, closing, convex_hull_image
from skimage.segmentation import mark_boundaries


start_time = time.time()

# np.set_printoptions(threshold=np.nan)

parser = OptionParser()
parser.add_option("-i", "--input_image", dest="image", type="string", action="store")
parser.add_option("-o", "--output_directory", dest="output_dir", type="string", action="store")

class Segmentation:

	def __init__(self, image, output_dir):

		self.image = io.imread(image).astype(np.float64)
		self.output_dir = output_dir
		self.x = self.image.shape[0]
		self.y = self.image.shape[1]
		self.dimensions = self.image.shape[0] * self.image.shape[1]
		# self.im_grey = io.imread(image, as_grey=True)
		self.im_grey = ndimage.imread(image)[:,:,0]

	def rgb2lab(self):
		## convert rgb to lab space
		lab = color.rgb2lab(self.image)
		lab1 = lab[:,:,0]
		lab2 = lab[:,:,1]
		lab3 = lab[:,:,2]
		## extract lab intensity space corresponding to nuclei
		# nuclei = lab3
		nuclei = filters.gaussian(lab3, 1)
		## flatten lab channels 
		lab1 = np.reshape(lab[:,:,0], (1, self.dimensions))
		lab2 = np.reshape(lab[:,:,1], (1, self.dimensions))
		lab3 = np.reshape(lab[:,:,2], (1, self.dimensions))
		## stack lab channels into one array
		lab_flattened = np.vstack((scale(lab1, axis=1),scale(lab2, axis=1),scale(lab3, axis=1)))
		return nuclei, lab_flattened

	def extract_gabor(self):

		im_grey, lab_features = S.rgb2lab()
		frequencies = [1.571]
		orientation_rads = [0, 0.785398, 1.5708, 2.35619]
		## generate list of combinations
		combos = list(IT.product(frequencies, orientation_rads))
		list_of_mags = []
		for item in combos:
			real_gabor, imaginary_gabor = filters.gabor(im_grey, item[0], item[1])
			## find the magnitude (square root of the sum of squares of real and imaginary gabor)
			mag = np.sqrt(np.square(real_gabor, dtype=np.float64)+np.square(imaginary_gabor, dtype=np.float64))
			sigma = .5 * ((2*pi)/item[0])
			K = 2
			## apply gaussian filter and scale the result with zero mean and unit variance
			mag_gaussian = filters.gaussian(mag, K*sigma)
			mag_gaussian_flattened = np.reshape(mag_gaussian, (1, (self.x*self.y)))
			list_of_mags.append(scale(mag_gaussian_flattened, axis=1))
		list_of_mags = np.asarray(list_of_mags).reshape(len(combos),self.dimensions)
		## combine gabor features with lab features
		im_grey_flattened = np.reshape(im_grey, (1, (self.x*self.y)))
		features = np.vstack((im_grey_flattened, list_of_mags))
		return features

	def kmeans(self):
		## performs KMeans on features, returns labels
		features = S.extract_gabor().T
		mkb = MiniBatchKMeans(n_clusters=3)
		mkb.fit(features)
		return mkb.labels_

	def visualize(self):

		labels = S.kmeans()
		labels = labels.reshape(self.x, self.y)
		intensity_list = []
		final_list = []
		final = self.im_grey
		for i in range(3):
			## mask grayscale image with labels 0 through 2
			im_grey = self.im_grey.copy()
			im_grey[labels != i] = 0
			## record mean intensities throughout labeled images
			final_list.append(im_grey)
			intensity_list.append(np.mean(im_grey))
			plt.imsave("./testing"+str(i)+".png", im_grey)
		min_index = intensity_list.index(min(intensity_list))
		im = final_list[min_index]
		## create and save binary mask
		# im_grey = np.zeros((self.x, self.y))
		# im_grey[im != 0] = 1
		# plt.imsave("./testing1_mb.png", im_grey, cmap="gray")
		## create and save grayscale mask
		final[im == 0] = 0
		labels, _ = ndimage.label(final)
		final_removed = remove_small_objects(labels, min_size=30)
		final[final_removed==0] = 0
		# plt.imsave("./objsrm.png", final, cmap="gray")
		return final

	def labels(self):
		img, _ = S.visualize()
		labels = label(img)
		measurements = regionprops(labels, self.im_grey)
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
		pickle.dump(meas_df, open("./feature_df.p", 'wb'))

	def watershed(self, image):
		im = S.visualize()
		inv = abs(255 - im)
		inv[inv == 255] = 0
		selem = disk(3)
		eroded = erosion(im, selem)
		w_tophat = white_tophat(inv, selem)


		fig, ax = plt.subplots(1, 3, figsize=(8, 4), sharex=True, sharey=True, subplot_kw={'adjustable': 'box-forced'})
		ax1, ax2, ax3 = ax.ravel()
		ax1.imshow(im, cmap="gray")
		# ax1.set_title('seg')
		ax2.imshow(inv, cmap="gray")
		# ax2.set_title('original_ndimage')
		ax3.imshow(w_tophat, cmap="gray")
		# ax3.set_title('original_io')
		# ax4.imshow(w_tophat, cmap="gray")

		plt.show()


		# plt.imsave("./watershed_386", a)




(options, args) = parser.parse_args()

S = Segmentation(options.image, options.output_dir)
# S.rgb2lab()
# S.extract_gabor()
# S.kmeans()
# S.visualize()
# S.save_label_matrix()
S.watershed(options.image)
print time.time() - start_time


