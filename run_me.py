######################################################################
#  CS589 Machine Learning 					HW5 Unsupervised Learning
#
#  Yi Fung   12/11/2017
######################################################################

import numpy as np
from scipy import misc
import imageio
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def read_scene():
	data_x = misc.imread('../../Data/Scene/times_square.jpg')
	return (data_x)

def read_faces():
	nFaces = 100
	nDims = 2500
	data_x = np.empty((0, nDims), dtype=float)
	for i in np.arange(nFaces):
		#data_x = np.vstack((data_x, np.reshape(misc.imread('../../Data/Faces/face_%s.png' % (i)), (1, nDims))))
		data_x = np.vstack((data_x, np.reshape(imageio.imread('../../Data/Faces/face_%s.png' % (i)), (1, nDims))))
	return (data_x)

def kmeans_compression(input_image, num_clusters):
	''' Helper function for kmeans '''
	kmeans = KMeans(n_clusters = num_clusters)
	kmeans.fit(input_image)
	labels = np.asarray(kmeans.labels_, dtype=np.uint8)
	clusters = np.asarray(kmeans.cluster_centers_, dtype=float)
	return np.array(list(map(lambda x: clusters[x], labels))), labels, clusters

def helper_plot_grid(cache, title, c=None):
	'''
	Helper function for plotting 3x3 grid, image visualization 
	'''
	f, axarr = plt.subplots(3, 3)
	axarr[0, 0].imshow(cache[0][0], cmap=c)
	axarr[0, 0].set_title(cache[0][1], fontsize=7)
	axarr[0, 1].imshow(cache[1][0], cmap=c)
	axarr[0, 1].set_title(cache[1][1], fontsize=7)
	axarr[0, 2].imshow(cache[2][0], cmap=c)
	axarr[0, 2].set_title(cache[2][1], fontsize=7)
	axarr[1, 0].imshow(cache[3][0], cmap=c)
	axarr[1, 0].set_title(cache[3][1], fontsize=7)
	axarr[1, 1].imshow(cache[4][0], cmap=c)
	axarr[1, 1].set_title(cache[4][1], fontsize=7)
	axarr[1, 2].imshow(cache[5][0], cmap=c)
	axarr[1, 2].set_title(cache[5][1], fontsize=7)
	axarr[2, 0].imshow(cache[6][0], cmap=c)
	axarr[2, 0].set_title(cache[6][1], fontsize=7)
	axarr[2, 1].imshow(cache[7][0], cmap=c)
	axarr[2, 1].set_title(cache[7][1], fontsize=7)
	axarr[2, 2].imshow(cache[8][0], cmap=c)
	axarr[2, 2].set_title(cache[8][1], fontsize=7)
	plt.setp([a.get_xticklabels() for a in axarr[0, :]], visible=False)
	plt.setp([a.get_xticklabels() for a in axarr[1, :]], visible=False)
	plt.setp([a.get_xticklabels() for a in axarr[2, :]], visible=False)
	plt.setp([a.get_yticklabels() for a in axarr[:, 0]], visible=False)
	plt.setp([a.get_yticklabels() for a in axarr[:, 1]], visible=False)
	plt.setp([a.get_yticklabels() for a in axarr[:, 2]], visible=False)
	plt.suptitle(title)
	plt.savefig("../figures/" + title + ".png")
	plt.clf()

def helper_plot(data, title, xaxis_label, yaxis_label):
	'''
	Helper function for the elbow plot
	'''
	x = data[0]
	y = data[1]
	plt.plot(x, y, "b*-")
	plt.title(title)
	plt.xlabel(xaxis_label)
	plt.ylabel(yaxis_label)
	plt.savefig("../figures/" + title + ".png")


if __name__ == '__main__':

	################################################
	# PCA

	data_x = read_faces()
	print('X = ', data_x.shape)

	print('Implement PCA here ...')
	U, s, V = np.linalg.svd(data_x)

	cache = [(data_x[1].reshape(50,50), "Original")]
	for i in [3,5,10,30,50,100,150,300]:
		print("k = " + str(i))
		w_k = V[:i]
		X_projected = np.dot(data_x, w_k.T)
		X_recon = np.dot(X_projected, w_k)
		print("Reconstruction error: " + str(np.sqrt(np.mean((data_x-X_recon)**2))))
		print("Compression rate: " + str((X_projected.nbytes + w_k.nbytes)/data_x.nbytes))
		cache.append((X_recon[1].reshape(50,50), "k=" + str(i)))
	helper_plot_grid(cache, "PCA reconstruction", "gray")

	################################################
	# K-Means

	data_x = read_scene()
	print('X = ', data_x.shape)

	print('Implement k-means here ...')

	cache = [(data_x,"Original")]
	cache2 = [[],[]]
	for i in [2,5,10,25,50,75,100,200]:
		print(str(i) + " clusters")
		flattened_image = data_x.ravel().reshape(data_x.shape[0] * data_x.shape[1], data_x.shape[2])
		print('Flattened image = ', flattened_image.shape)
		flattened_image, labels, clusters = kmeans_compression(flattened_image, i)
		reconstructed_image = flattened_image.ravel().reshape(data_x.shape[0], data_x.shape[1], data_x.shape[2])
		print('Reconstructed image = ', reconstructed_image.shape)
		print("Reconstruction error: " + str(np.sqrt(np.mean((data_x-reconstructed_image)**2))))
		print("Compression rate: " + str((clusters.nbytes + 400*400*np.ceil(np.log2(i))/8.0) / data_x.nbytes))
		cache.append((reconstructed_image/255, str(i) + " clusters"))
		cache2[0].append(i)
		cache2[1].append(np.sum(np.square(data_x-reconstructed_image))**0.5)
	helper_plot_grid(cache, "KMeans reconstruction")
	helper_plot(cache2, "Elbow Plot for KMeans Clustering", "K", "Sum of Pixel Squared Errors")