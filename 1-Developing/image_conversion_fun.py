## CODE FROM https://alaska.utt.fr/

from PIL import Image
import numpy as np
from scipy.ndimage import filters
from scipy.signal import medfilt
from scipy.ndimage import filters
from scipy.signal import medfilt
from matplotlib import pyplot as plt
import tifffile


# **************************#
# MERE (center) crop #
# **************************#
def center_crop(im, newWidth, newHeight):
    width, height = im.shape[0:2]  # Get dimensions
    if newWidth > width or newHeight > height:
        newWidth, newHeight = newHeight, newWidth
    
# Here we simply compute the first and last indices of pixels' central area.
    left = (width - newWidth) // 2
    top = (height - newHeight) // 2
    right = (width + newWidth) // 2
    bottom = (height + newHeight) // 2
# and merely return the pixels from this area ...
    return(im[left:right, top:bottom,:])


# **************************#
# RGB2GRAY (used when using "smart crop" #
# **************************#
def rgb2gray(rgb):
# Here we use the good old standard ITU-R Recommendation BT.601 (rec601) for computing luminance: 
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return(gray)


# **************************#
# Mere JPEG compression function #
# **************************#
def jpeg_compression(infile, outpath, qf):
    if (infile.endswith(".tiff") or infile.endswith(".tif")) and (outpath.endswith(".jpg") or outpath.endswith(".jpeg")):
        try:
            Image.open(infile).save(outpath, quality=qf, subsampling=0)
        except IOError:
            print("Cannot convert {} to {}".format(infile, outpath))
    else:
        print("Non TIFF image source OR Non JPG image target ... convertion stopped ...")


# **************************#
# MOST complex function for resizing (can either by crop / resize with resampling or both) #
# **************************#
def image_randomizeResizing(infile, outpath, new_width, new_height, subsampling_type=0, kernel=Image.LANCZOS, resize_weight=0.5, resize_factor_UB=1.25):
# We used three subsampling_type :  0 -> resize and crop ;  1 -> resize (by resampling) only ; 2 -> crop only
# when using both ( subsampling_type == 0 ) one must set the amount of each; to this end we used the variable from random_generator developement resize_weight
# The principle is to compute first the minimal resampling factor (when used alone) and then sample between this value and 1.25 (corresponding to upsampling by 25%)

	if (infile.endswith(".tiff") or infile.endswith(".tif")) and (outpath.endswith(".tiff") or outpath.endswith(".tif")):
		im = (tifffile.imread(infile)/(2**16-1))
	# GRID size is the shift used to evaluated the content of each patch in order to select the one with most content. 
		GRIDSIZE=64
	# Doing resize and then crop. 
		if subsampling_type == 0:
	# For this we randomly select first the resizing factor.
	# This is drawn from a (uniform) distribution between the minimal factor (in order not to have a resized image of smaller dimension than the targeted size) up to 1.25 which corresponds to upsize by 25%
	# There we compute first the minimal resizing factor
			resize_FactorMin = max( float( new_width )/im.shape[0] , float( new_height )/im.shape[1] )
	# Then we generate a random value Uniformly in the range [ 0 ;  1 ] and scale it to the acceptable range [ MIN ; 1.25 ]
			resize_Factor = resize_FactorMin + ( resize_weight ) * ( resize_factor_UB - resize_FactorMin )
	# Then we carry out the resizing to the dimension multilplied by scaling factor
			temp_width = int(round( im.shape[0] * resize_Factor ) )
			temp_height = int(round( im.shape[1] * resize_Factor ) ) 
			im = resize_keep_aspect(im,  temp_width , temp_height , kernel)
	# Then we do the smart crop ; to avoid any issue, we always check before that the image size allows having minimal number of "GRID" / patch otherwise there is a high risk not to have any relevant patch
			if GRIDSIZE > min( temp_width - new_width , temp_height - new_height )/2:
				im = center_crop(im, new_width, new_height)
			else:
				im = edge_crop(im, 1.5, new_height, new_width,GRIDSIZE)

	# Doing resize (by resampling) alone. 
		elif subsampling_type == 1:
	# The resizing factor is computed only for logging purpose
			resize_Factor = max( new_width/im.shape[0] , new_height/im.shape[1] )
			im = resize_keep_aspect(im, new_width, new_height, kernel)
			im = center_crop(im, new_width, new_height)

	# Doing mere crop
		elif subsampling_type ==2:
			if GRIDSIZE > min( im.shape[0] - new_width , im.shape[1] - new_height )/2:
				im = center_crop(im, new_width, new_height)
			else:
				im = edge_crop(im, 1.5, new_height, new_width,GRIDSIZE)
			resize_Factor = 0

	# Writting image after resizing
		im = np.round(im*(2**16-1))
		im[im > 2**16-1] = 2**16-1
		im[im < 0] = 0
		im = im.astype(np.uint16)
		tifffile.imwrite(outpath, im)

		return resize_Factor


# **************************#
# Resizing function while ensuring keeping the aspect ration #
# **************************#
def resize_keep_aspect(im, new_w, new_h, kernel):
    width, height = im.shape[0:2]
    width = float( width )
    height = float( height )
    aspect = width/height
# Here we compute the maximal resizing factor with respect to both dimension
    resize = max( float(new_w) /  width  , float(new_h) / height )
# and then adjust the  new_width and new_height
    new_w = round( width * resize )
    new_h = round( height * resize )

    # resizing and writing out the resulting image ;
    res_im = np.zeros((new_w,new_h,3))
    res_im[:,:,0] = np.array(Image.fromarray(im[:,:,0]).resize((new_h, new_w), kernel))
    res_im[:,:,1] = np.array(Image.fromarray(im[:,:,1]).resize((new_h, new_w), kernel))
    res_im[:,:,2] = np.array(Image.fromarray(im[:,:,2]).resize((new_h, new_w), kernel))
    max_val = np.max([res_im[:,:,0].max(), res_im[:,:,1].max(), res_im[:,:,2].max()])
    if max_val > 1:
        res_im[:,:,0] = res_im[:,:,0]/max_val 
        res_im[:,:,1] = res_im[:,:,1]/max_val
        res_im[:,:,2] = res_im[:,:,2]/max_val
    
    return(res_im)


# **************************#
# SMART crop function, that selects the area with most content #
# **************************#
# In brief, it is based on a wavelet decompositiotn (app for approximation while det stand for details)
# and we compute edges based on approximations ; while details are used to adjust the threshold wrt the image noise
# Original method from A Foi, M Trimeche, V Katkovnik, K Egiazarian, "Practical Poissonian-Gaussian noise modeling and fitting for single-image raw-data", IEEE Transactions on Image Processing 17 (10), 1737-1754
def edge_crop(Z, threshold, cropH, cropW,grid):
    # Conversion of image into grayscale
    if Z.shape[2] ==3:
        X = rgb2gray(Z)
    elif Z.shape[2] ==1: # Is Z is already grayscale, let us keep it unchanged
        X = Z
    # definition of filters' Kernel
    unif_kernel = (1/49) * np.ones([7,7])
    gradient_kernel = np.matrix(' -1 0 1; -2 0 2 ; -1 0 1')
    laplacian_kernel = (1/112)*np.matrix(""" 0 0 0 1 0 0 0 ;
                             0 0 3 -12 3 0 0 ; 
                             0 3 -24 57 -24 3 0 ;
                             1 -12 57 -112 57 -12 1;
                             0 3 -24 57 -24 3 0 ;
                             0 0 3 -12 3 0 0 ; 
                             0 0 0 1 0 0 0""")
    
    # Wavelet  definitions
    psi_1 = np.array([0.035, 0.085, -0.135, -0.460, 0.807, -0.333])
    psi = psi_1[:, np.newaxis] * psi_1[np.newaxis, :]
    
    phi_1 = np.array([0.025, -0.060, -0.095, 0.325, 0.571 ,0.235])
    phi = phi_1[:, np.newaxis] * phi_1[np.newaxis, :]
    
    # Once those parameters are defined, compute "approximations" and "detail"
    z_app = filters.convolve(X, phi)
    z_det = filters.convolve(X, psi)

    s = np.sqrt(np.pi/2) * filters.convolve(np.abs(z_det), unif_kernel)
    
    lam = filters.convolve(medfilt(z_app), laplacian_kernel)
    
    # Edge detection via differentiation filtering along horizontal and vertical direction
    vert_grad = filters.convolve(lam, gradient_kernel)
    horz_grad = filters.convolve(lam, gradient_kernel.T)
    edge_detector = np.abs(vert_grad)  + np.abs(horz_grad) + np.abs(lam)
    
    X_edge = edge_detector > s*threshold
    
    # Once edge detection has been carried out; select the area with highest number of edges
    candidates =       np.array([(x, y) for x,y in zip(np.arange(0, (Z.shape[0]-cropH), grid),np.arange(0, (Z.shape[1]-cropW),grid))])
    candidates_score = np.array([np.sum(X_edge[x:x+cropH, y:y+cropW]) for x,y in zip(np.arange(0, (Z.shape[0]-cropH), grid),np.arange(0, (Z.shape[1]-cropW),grid))])
    if candidates_score.shape[0]==0:
        Z2 = center_crop(Z, cropH, cropW)
        return(Z2)
    else:
        best_idx = np.argmax(candidates_score)
        return(Z[candidates[best_idx][0]:candidates[best_idx][0]+cropH, candidates[best_idx][1]:candidates[best_idx][1]+cropW, :] )
