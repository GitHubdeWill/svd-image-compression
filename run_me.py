########################################################################
#
#  Math 545 Final Project
#
#  Professor: Qian-Yong Chen
#  Students: Yi Fung, William He
#
#  Due Date: April 30, 2018
#
#  Note: this is file that actually perform SVD and image compression.
#
#  Thank you for grading and have a great summer!
#
########################################################################



### Import libraries at the top ###
import numpy as np
from scipy import misc
import imageio
from PIL import Image
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def read_faces():
	'''
	Helper function to read in the image data

	Input: None
	Output: data_x - a 20 x 10000 matrix
	'''
	nFaces = 20+1
	nDims = 100*100
	data_x = np.empty((0, nDims), dtype=float)
	for i in np.arange(1, nFaces):
		data_x = np.vstack((data_x, np.reshape(imageio.imread('./pics/%s.jpeg' % (i)), (1, nDims))))
	return (data_x)


def read_faces_rgb():
	'''
	Helper function to read in the image data in colored format

	Input: None
	Output: data_x for R, data_x for G, data_x for B
	'''
	### May be not right!!! ###
	nFaces = 20+1
	nDims = 100*100
	data_x_r = np.empty((0, nDims), dtype=float)
	data_x_g = np.empty((0, nDims), dtype=float)
	data_x_b = np.empty((0, nDims), dtype=float)
	for i in np.arange(1, nFaces):
		im = Image.open('./pics2/%sRGB.jpeg' % (i)).convert('RGB')
		pix = np.array(im)
		data_x_r = np.vstack((data_x_r, np.reshape(pix[:,:,0], (1,nDims))))
		data_x_g = np.vstack((data_x_g, np.reshape(pix[:,:,1], (1,nDims))))
		data_x_b = np.vstack((data_x_b, np.reshape(pix[:,:,2], (1,nDims))))
	return (data_x_r, data_x_g, data_x_b)


def helper_plot_grid(cache, title, c=None):
	'''
	Helper function for plotting 3x3 grid, image visualization

	Input: cache data of one the images using PCA w/ different # of components
	Output: Outputs the 3x3 plot and returns nothing 
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

	########## This is the driver and main function ############
	
	print('Running PCA...')

	data_x = read_faces()
	print('Size of X = ', data_x.shape)

	### Perform SVD decomposition here
	U, s, V = np.linalg.svd(data_x)
	#print('Size of U = ', U.shape)
	#print('Size of s = ', s.shape)
	#print('Size of V = ', V.shape)

	cache = [(data_x[1].reshape(100,100), "Original")]

    ### PCA: find the k=i eigenvectors corresponding to the largest
    ###      eigenvalues in the SVD-decomposed data 
	for i in [3,5,10,25,50,100,150,200]:
		print("k = " + str(i))
		w_k = V[:i]
		### Report the compression rate here
        X_projected = np.dot(data_x, w_k.T)
		#print("# bytes in X_projected: " + str(X_projected.nbytes))
		#print("# bytes in w_k: " + str(w_k.nbytes))
		#print("# bytes in data_x: " + str(data_x.nbytes))
		print("Compression rate: " + str(float(X_projected.nbytes + w_k.nbytes)/float(data_x.nbytes)))
		### Report the reconstruction error here
		X_recon = np.dot(X_projected, w_k)
		print("Reconstruction error: " + str(np.sqrt(np.mean((data_x-X_recon)**2))))
		### Cache data for plotting 
		cache.append((X_recon[1].reshape(100,100), "k=" + str(i)))

	### Call on helper function to plot results	
	helper_plot_grid(cache, "PCA reconstruction", "gray")
	
	####### PCA for colore images #######
	print('Running PCA, colored version...')

	data_x_r, data_x_g, data_x_b = read_faces_rgb()
	U_r, s_r, V_r = np.linalg.svd(data_x_r)
	U_g, s_g, V_g = np.linalg.svd(data_x_g)
	U_b, s_b, V_b = np.linalg.svd(data_x_b)

	coloredIm = np.dstack((data_x_r[0],data_x_g[0],data_x_b[0])).reshape(100,100,3)
	
	cache = [(coloredIm, "Original")]
	for i in [5,10,20,30,40,50,60,70,100]:
		print("k = " + str(i))
		w_k_r = V_r[:i]
		X_projected_r = np.dot(data_x_r, w_k_r.T)
		X_recon_r = np.dot(X_projected_r, w_k_r)
		w_k_g = V_g[:i]
		X_projected_g = np.dot(data_x_g, w_k_g.T)
		X_recon_g = np.dot(X_projected_g, w_k_g)
		w_k_b = V_b[:i]
		X_projected_b = np.dot(data_x_b, w_k_b.T)
		X_recon_b = np.dot(X_projected_b, w_k_b)
		#print("Reconstruction error: " + str(np.sqrt(np.mean((data_x-X_recon)**2))))
		#print("# bytes in X_projected: " + str(X_projected.nbytes))
		#print("# bytes in w_k: " + str(w_k.nbytes))
		#print("# bytes in data_x: " + str(data_x.nbytes))
		#print("Compression rate: " + str(float(X_projected.nbytes + w_k.nbytes)/float(data_x.nbytes)))
		cache.append((np.dstack((X_recon_r[0],X_recon_g[0],X_recon_b[0])).reshape(100,100,3), "k=" + str(i)))

	helper_plot_grid(cache, "PCA reconstruction - colored image")
	
