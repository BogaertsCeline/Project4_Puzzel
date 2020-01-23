import numpy as np
import scipy
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
import matplotlib as mpl
import matplotlib.pyplot as plt
from functools import partial
import cv2
import skimage
from sklearn.cluster import KMeans
from utils import get_line_through_points, distance_point_line_squared, distance_point_line_signed, rotate
import logging

logging.basicConfig(level=logging.INFO)

_corner_indexes = [(0, 1), (1, 3), (3, 2), (0, 2)]


def compute_minmax_xy(thresh):
	"""
	Given the thresholded image, compute the minimum and maximum x and y
	coordinates of the segmented puzzle piece.
	"""
	idx_shape = np.where(thresh == 0)
	return [np.array([coords.min(), coords.max()]) for coords in idx_shape]


def segment_piece(image, bin_threshold=128):

	"""
	Apply segmentation of the image by simple binarization
	"""
	return cv2.threshold(image, bin_threshold, 255, cv2.THRESH_BINARY)[1]


def extract_piece(thresh , original):
	# deze nodig

	# Here we build a square image centered on the blob (piece of the puzzle).
	# The image is constructed large enough to allow for piece rotations.
	minmax_y, minmax_x = compute_minmax_xy(thresh)

	ly, lx = minmax_y[1] - minmax_y[0], minmax_x[1] - minmax_x[0]
	size = max(ly, lx)+1

	x_extract = thresh[minmax_y[0]:minmax_y[1] + 1, minmax_x[0]:minmax_x[1] + 1]
	x_extract_3D = cv2.merge((x_extract, x_extract, x_extract))

	ly, lx = x_extract.shape

	x_copy = np.full((size, size), 255, dtype='uint8')
	x_copy_3D = cv2.merge((x_copy, x_copy, x_copy))

	sy, sx = size // 2 - ly // 2, size // 2 - lx // 2

	x_copy[sy: sy + ly, sx: sx + lx] = x_extract

	x_copy_3D[sy: sy + ly, sx: sx + lx] = x_extract_3D

	punt_x = minmax_x[0] - ((size - lx) // 2)
	punt_y = minmax_y[0] - ((size - ly) // 2)

	if punt_y <0:
		x_copy= x_copy[abs(punt_y):size, 0: size]

		x_copy = cv2.copyMakeBorder(x_copy,
									top = 0,
									bottom=abs(punt_y),
									left=0,
									right=0,
									borderType=cv2.BORDER_CONSTANT,
									value=[255,255,255] )
		punt_y= 0

	if punt_x <0:
		x_copy= x_copy[0:size, abs(punt_x): size]

		x_copy = cv2.copyMakeBorder(x_copy,
									top = 0,
									bottom=0,
									left=abs(punt_x),
									right=0,
									borderType=cv2.BORDER_CONSTANT,
									value=[255,255,255] )
		punt_x= 0

	cropped_original = original[punt_y:punt_y+size, punt_x:punt_x+size]

	thresh = x_copy
	thresh = 255 - thresh

	thresh_3D = x_copy_3D
	thresh_3D = 255 - thresh_3D


	cropped_original[thresh == 0] = [0, 0, 0]

	return [thresh, thresh_3D, cropped_original]


####################################################################################################################
		

def get_default_params():
	#deze nodig
	side_extractor_default_values = {
		'before_segmentation_func': partial(cv2.medianBlur, ksize=5),
		'bin_threshold': 130,
		'after_segmentation_func': None,
		'scale_factor': 0.5,
		'harris_block_size': 5,
		'harris_ksize': 5,
		'corner_nsize': 5,
		'corner_score_threshold': 0.2,
		'corner_minmax_threshold': 100,
		'corner_refine_rect_size': 5,
		'edge_erode_size': 3,
		'shape_classification_distance_threshold': 100,
		'shape_classification_nhs': 5,
		'inout_distance_threshold': 5
	}
		
	return side_extractor_default_values.copy()


def process_piece(image, **kwargs):

	params = get_default_params()
	for key in kwargs:
		params[key] = kwargs[key]
		
	out_dict = {}
	try:
		original = image.copy()
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

		before_segmentation_func = params['before_segmentation_func']
		after_segmentation_func = params['after_segmentation_func']
		bin_threshold = params['bin_threshold']


		if before_segmentation_func:
			gray = before_segmentation_func(gray)

		gray = segment_piece(gray, bin_threshold)

		if after_segmentation_func:
			gray = after_segmentation_func(gray)

		gray = 255 - gray

		ret, labels = cv2.connectedComponents(gray)

		connected_areas = [np.count_nonzero(labels == l) for l in range(1, ret)]
		max_area_idx = np.argmax(connected_areas) + 1
		gray[labels != max_area_idx] = 0
		gray = 255 - gray
		out_dict['mask'],out_dict['mask_3D'], out_dict['cropped_original'] = extract_piece(gray, original)

	except Exception as e:
		out_dict['error'] = e
		
	finally:
		return out_dict

