'''
File to measure and print SSIM and PSNR and inception score of images
'''

import os
import ntpath
import numpy as np
from scipy import misc
from utilityFiles.data_utils import getPaths
from utilityFiles.ssm_psnr_inceptionscore_utils import getSSIM, getPSNR
from sklearn.metrics import f1_score
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import os.path
import sys
import tarfile
from numpy.random import shuffle
import torchvision.transforms as transforms
import numpy as np
from six.moves import urllib
import tensorflow as tf
import glob
import scipy.misc
import math
import sys
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy.random import random
from scipy.linalg import sqrtm
from math import floor
from numpy import ones
from numpy import expand_dims
from numpy import log
from numpy import mean
from numpy import std
from numpy import exp
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from numpy.random import shuffle
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.datasets import cifar10
from skimage.transform import resize
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data

from torchvision.models.inception import inception_v3

import numpy as np
from scipy.stats import entropy
from numpy import asarray
from math import floor
from numpy import ones
from numpy import expand_dims
from numpy import log
from numpy import mean
from numpy import std
from numpy import exp
from numpy.random import shuffle
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.datasets import cifar10
from skimage.transform import resize
from numpy import asarray
 
## data paths
GEN_im_dir  = "data/outputFinalImage/"  
GTr_im_dir  = 'data/test/GroundTruth/' 


'''
Function to calclaute ssim and psnr of model 
'''
## compares avg ssim and psnr 
def measure_SSIM_PSNRs(GT_dir, Gen_dir):
    
    GT_paths, Gen_paths = getPaths(GT_dir), getPaths(Gen_dir)
    ssims, psnrs,inceps = [], [],[]
    for img_path in GT_paths:
        name_split = ntpath.basename(img_path).split('.')
        gen_path = os.path.join(Gen_dir, name_split[0]+'_gen.png') #+name_split[1])
        if (gen_path in Gen_paths):
            r_im = misc.imread(img_path)
            g_im = misc.imread(gen_path)
            assert (r_im.shape==g_im.shape), "The images should be of same-size"
            ssim = getSSIM(r_im, g_im)
            psnr = getPSNR(r_im, g_im)
            incep = calculate_inception_score(g_im)
            ssims.append(ssim)
            psnrs.append(psnr)
            inceps.append(incep)
    return np.array(ssims), np.array(psnrs),np.array(inceps)

#scale images
def scale_images(images, new_shape):
	images_list = list()
	for image in images:
		# resize with nearest neighbor interpolation
		new_image = resize(image, new_shape, 0)
		# store
		images_list.append(new_image)
	return asarray(images_list)
 
# calulate inception score
def calculate_inception_score(images, n_split=10, eps=1E-16):
	# load inception v3 model
	model = InceptionV3()
	# enumerate splits of images/predictions
	scores = list()
	n_part = floor(images.shape[0] / n_split)
	for i in range(n_split):
		# retrieve images
		ix_start, ix_end = i * n_part, (i+1) * n_part
		subset = images[ix_start:ix_end]
		# convert from uint8 to float32
		subset = subset.astype('float32')
		# scale images to the required size
		subset = scale_images(subset, (299,299,3))
		# pre-process images, scale to [-1,1]
		subset = preprocess_input(subset)
		# predict p(y|x)
		p_yx = model.predict(subset)
		# calculate p(y)
		p_y = expand_dims(p_yx.mean(axis=0), 0)
		# calculate KL divergence using log probabilities
		kl_d = p_yx * (log(p_yx + eps) - log(p_y + eps))
		# sum over classes
		sum_kl_d = kl_d.sum(axis=1)
		# average over images
		avg_kl_d = mean(sum_kl_d)
		# undo the log
		is_score = exp(avg_kl_d)
		# store
		scores.append(is_score)
	# average across images
	is_avg, is_std = mean(scores), std(scores)
	return scores

### compute SSIM and PSNR
SSIM_measures, PSNR_measures,INCEP_measures = measure_SSIM_PSNRs(GTr_im_dir, GEN_im_dir)
print ("SSIM of model >> Mean: {0} std: {1}".format(np.mean(SSIM_measures), np.std(SSIM_measures)))
print ("PSNR of model >> Mean: {0} std: {1}".format(np.mean(PSNR_measures), np.std(PSNR_measures)))
print ("Inception score of model >> Mean: {0} std: {1}".format(np.mean(INCEP_measures), np.std(INCEP_measures)))





