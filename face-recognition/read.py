import matplotlib.pyplot as plt
import msgpack
import numpy as np

import sys
sys.path.append('msgpack-numpy')
import msgpack_numpy as m
m.patch()

filenames = [
	'data/face_images/face_images1.bin',
	'data/face_images/face_images2.bin',
	'data/face_images/face_images3.bin',
	'data/face_images/face_images4.bin',
	'data/face_images/face_images5.bin',
]

data_images = []
data_coords = []

for filename in filenames:
	with open(filename, 'rb') as data_file:
		main_dict = msgpack.load(data_file)
	
	images_i = main_dict[b'images']
	coords_i = main_dict[b'co-ords']
	
	data_images.append(images_i)
	data_coords.append(coords_i)


images = np.vstack(data_images)
coords = np.vstack(data_coords)

faces = []
eyes_l = []
eyes_r = []

for image, coords in zip(images, coords):
	face_coords = coords[:4]
	eye_r_coords = coords[4:8]
	eye_l_coords = coords[8:]
	left_face, top_face, height_face, width_face = face_coords
	left_eye_r, top_eye_l, height_eye_l, width_eye_l = eye_l_coords
	left_eye_r, top_eye_r, height_eye_r, width_eye_r = eye_r_coords
	
	cropped_face = image[
		top_face:top_face+height_face,
		left_face:left_face+width_face
	]
	cropped_eye_l = image[
		top_eye_l:top_eye_l+height_eye_l,
		left_eye_l:left_eye_l+width_eye_l
	]
	cropped_eye_r = image[
		top_eye_r:top_eye_r+height_eye_r,
		left_eye_r:left_eye_r+width_eye_r
	]
	
	plt.imshow(image)
	
	plt.figure()
	plt.imshow(cropped_face)
	
	plt.figure()
	plt.imshow(cropped_eye_l)
	
	plt.figure()
	plt.imshow(cropped_eye_r)
	
	faces.append(cropped_face)
	eyes_l.append(cropped_eye_l)
	eyes_r.append(cropped_eye_r)
	
	plt.show()
