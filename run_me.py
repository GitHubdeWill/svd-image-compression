import numpy as np
from scipy import misc
import imageio
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def read_faces():
	nFaces = 21
	nDims = 10000
	data_x = np.empty((0, nDims), dtype=float)
	for i in np.arange(1, nFaces):
		#data_x = np.vstack((data_x, np.reshape(misc.imread('../../Data/Faces/face_%s.png' % (i)), (1, nDims))))
		data_x = np.vstack((data_x, np.reshape(imageio.imread('./pics/%s.jpeg' % (i)), (1, nDims))))
	return (data_x)


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
	plt.savefig("./figures/" + title + ".png")
	plt.clf()

if __name__ == '__main__':

	################################################
	# PCA

	data_x = read_faces()
	print('X = ', data_x.shape)

	print('Implement PCA here ...')
	U, s, V = np.linalg.svd(data_x)
	print(V.shape)

	cache = [(data_x[1].reshape(100,100), "Original")]
	for i in [3,5,10,25,50,100,150,200]:
		print("k = " + str(i))
		w_k = V[:i]
		X_projected = np.dot(data_x, w_k.T)
		X_recon = np.dot(X_projected, w_k)
		print("Reconstruction error: " + str(np.sqrt(np.mean((data_x-X_recon)**2))))
		print(X_projected.nbytes)
		print(w_k.nbytes)
		print(data_x.nbytes)
		print("Compression rate: " + str(float(X_projected.nbytes + w_k.nbytes)/float(data_x.nbytes)))
		cache.append((X_recon[1].reshape(100,100), "k=" + str(i)))
	helper_plot_grid(cache, "PCA reconstruction", "gray")
