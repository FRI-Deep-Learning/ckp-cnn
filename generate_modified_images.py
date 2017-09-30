import numpy as np
import os
import cv2
from keras.preprocessing.image import ImageDataGenerator, load_img
from tqdm import tqdm

gen = ImageDataGenerator(
							rotation_range=30,			#range that pictures can be rotated
							width_shift_range=0.2,		#fraction of total width that randomly translates pictures horizontally
							height_shift_range=0.2,		#fraction of total height that randomly translates pictures vertically
							zoom_range=0.3,				#range for random zoom
							horizontal_flip=True,		#randomly flips inputs horizontally
							vertical_flip=True,			#randomly flips inputs vertically
							cval =255,					#sets the fill color
							fill_mode='constant')		#fills in background after image transformation					
							
dir = "cohn-kanade-images"


for subdir in os.listdir(dir):
	for root, dirs, files in os.walk(os.path.join(dir, subdir)):
		for file in tqdm(files):
			if file.endswith(".png"):
				img = load_img(os.path.join(root, file))
				arr = np.array(img)
				arr = arr.reshape((1,)+arr.shape)
				for i in range(10):				
					for batch in gen.flow(x=arr,batch_size=1,save_to_dir=root,save_prefix="m"+str(i)+"_"+file[:-4],save_format='png'):
						break