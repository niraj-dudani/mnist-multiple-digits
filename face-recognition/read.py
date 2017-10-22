import msgpack
import sys
sys.path.append('msgpack-numpy')

import msgpack_numpy as m
m.patch()

filename = 'data/face_images/face_images1.bin'

with open(filename, 'rb') as data_file:
	main_dict = msgpack.load(data_file)

coords = main_dict[b'co-ords']
images = main_dict[b'images']
