import numpy as np
import cv2
import gzip
import umsgpack
import random
from tqdm import tqdm

file = gzip.open("preprocessed_db.gz", "rb")
output_file = gzip.open("occluded_db.gz","wb")

num_images = umsgpack.unpack(file)
out_num_images= num_images*3;

map = umsgpack.unpack(file)
out_map = {}

for folder,num in map.items():
	out_map.update({folder : num * 3})

output_file.write(umsgpack.packb(out_num_images))
output_file.write(umsgpack.packb(out_map))

random.seed()

for (folder,num) in tqdm(map.items()):
	for i in range(0,num):
		
		(subject, image) = umsgpack.unpack(file)
		output_file.write(umsgpack.packb((subject,image)))
		im = np.asarray(image)
		#copies image so there are two separate images - one that contains upper occlusion & one that contains lower occlusion
		im2 = im.copy()
		#randomly choose dimensions ranging from 10-14x36-42
		h= random.randint(10,15)
		w= random.randint(36,43)
		rect = np.zeros((h,w))
		#randomly chooses a type of noise
		noise_type = random.randint(0,2)
		if noise_type == 0:
			#gaussian noise
			rmean = random.randint(20,131)
			cv2.randn(rect, rmean,30)
		else:
			#salt & pepper noise
			cv2.randu(rect,0,255)
		
		#places occluded rectangle on image in random location centered on the eye region 
		rx = random.randint(-3,4)
		ry = random.randint(-1,2)
		im[42+ry:42+h+ry, 14+rx:14+w+rx] = rect
		#upper-occluded image
		output_file.write(umsgpack.packb((subject,im.tolist())))
		#places occluded rectangle on image in random location centered on the mouth region 
		rx = random.randint(-3,4)
		ry = random.randint(-1,2)
		im2[19+ry:19+h+ry, 14+rx:14+w+rx] = rect
		#lower-occluded image
		output_file.write(umsgpack.packb((subject,im2.tolist())))
		
		
#mean_image
output_file.write(umsgpack.packb(umsgpack.unpack(file)))