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
                d = msgpack.load(data_file)

        images_i = d[b'images']
        coords_i = d[b'co-ords']

        data_images.append(images_i)
        data_coords.append(coords_i)

data_images = np.vstack(data_images)
data_coords = np.vstack(data_coords)

def crop(I,r):
        """ Return a cropped version of image 'I', 'r' being the
        cropping rectangle.
        Returns a view, please use numpy.copy if desired on the
        returned object.
        'r' is an array of length 4 representing - [top-left-x,
        top-left-y, width, height]
        'top' is defined with a flipped y-axis. That is, lower
        y-coordinate is on top of a higher y-coordinate."""
        c1 = r[0]
        c2 = r[1]
        w = r[2]
        h = r[3]
        return I[c2:c2+h,c1:c1+w]

def anti_crop(I,r,min_abs_offset=1):
        """Crops image I somewhere other than the specified rectangle.
        Same as crop, but does the cropping after translating the
        rectangle by a random offset. The minimum absolute value of
        the offset is specified by 'min_abs_offset'
        We assume sane values for crop rectangle and min_abs_offset."""
        c1 = r[0]
        c2 = r[1]
        w = r[2]
        h = r[3]
        H = I.shape[0]
        W = I.shape[1]
        ############################
        # pick new top-left corner #
        ############################
        # When picking the new top-left-corner, we leave out strips at
        # the edges, so that the new crop rectangle does not cross the
        # boundaries of the image. We also leave out a hole in the
        # middle, a square of size min_abs_offset.
        c1_new = np.random.randint(0, W - (w - 1) - 2 * min_abs_offset)
        # add the hole, due to min_abs_offset.
        if (c1_new > (c1 - min_abs_offset)):
                c1_new += 2 * min_abs_offset
        c2_new = np.random.randint(0, H - (h - 1) - 2 * min_abs_offset)
        # add the hole, due to min_abs_offset.
        if (c2_new > (c2 - min_abs_offset)):
                c2_new += 2 * min_abs_offset

        new_rect = [c1_new, c2_new, w, h]
        return crop(I, new_rect)

def translate_rect(r,t):
        """Translates rectangle r by t=[t1,t2]
        'r' is an array of length 4 representing - [top-left-x,
        top-left-y, width, height]
        'top' is defined with a flipped y-axis. That is, lower
        y-coordinate is on top of a higher y-coordinate."""
        c1 = r[0]
        c2 = r[1]
        w = r[2]
        h = r[3]
        t1 = t[0]
        t2 = t[1]
        return [c1 + t1, c2 + t2, w, h]
        
faces = []
eyes_l = []
eyes_r = []

non_faces = []
non_eyes_l = []
non_eyes_r = []

NON_FACE_IMAGES_PER_INPUT_IMAGE = 5
NON_EYE_L_IMAGES_PER_INPUT_IMAGE = 5
NON_EYE_R_IMAGES_PER_INPUT_IMAGE = 5
for image, coords in zip(data_images, data_coords):
        face_coords = coords[:4]
        eye_l_coords = coords[4:8]
        eye_r_coords = coords[8:12]

        if -1 not in face_coords:
                cropped_face_image = crop(image,face_coords)
                faces.append(cropped_face_image)
                for i in xrange(NON_FACE_IMAGES_PER_INPUT_IMAGE):
                        non_faces.append(anti_crop(image,face_coords, min_abs_offset=20))
                if -1 not in eye_l_coords:
                        for i in xrange(NON_EYE_L_IMAGES_PER_INPUT_IMAGE):
                                non_eyes_l.append(anti_crop(cropped_face_image,
                                                            translate_rect(eye_l_coords, face_coords[:2]),
                                                            min_abs_offset=5))
                if -1 not in eye_r_coords:
                        for i in xrange(NON_EYE_R_IMAGES_PER_INPUT_IMAGE):
                                non_eyes_r.append(anti_crop(cropped_face_image,
                                                            translate_rect(eye_r_coords, face_coords[:2]),
                                                            min_abs_offset=5))
                        
        if -1 not in eye_l_coords:
                cropped_eye_l = crop(image,eye_l_coords)
                eyes_l.append(cropped_eye_l)

        if -1 not in eye_r_coords:
                cropped_eye_r = crop(image,eye_r_coords)
                eyes_r.append(cropped_eye_r)



N=10
# show_what = non_faces
show_what = non_eyes_l
# show_what = non_eyes_r
for i in xrange(N*N):
        plt.subplot(N,N,i+1)
        plt.imshow(show_what[i])
        plt.axis('off')
        plt.title(str(i))
plt.show()
