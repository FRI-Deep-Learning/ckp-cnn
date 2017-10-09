import cv2
import numpy as np
import gzip
import umsgpack
from tqdm import tqdm
import os

db_dir = "cohn-kanade-images"
output_file = "preprocessed_db.gz"

out = gzip.open(output_file, "wb")

subjects = os.listdir(db_dir)

mean_image = np.zeros((64, 64), dtype = np.float32)

image_files = []

subject_image_map = {s: 0 for s in subjects}

print("> Finding subjects...")
for subject in tqdm(subjects):
    for root, dirs, files in os.walk(os.path.join(db_dir, subject)):
        for file in files:
            if file.startswith("face") and file.endswith(".png"):
                image_files.append((subject, os.path.join(root, file)))
                subject_image_map[subject] += 1

num_images = len(image_files)
umsgpack.pack(num_images, out)

umsgpack.pack(subject_image_map, out)

print("> Packaging images...")
for (subject, file) in tqdm(image_files):
    img = cv2.imread(file, 0)
    mean_image += img.astype(np.float32) / num_images
    out.write(umsgpack.packb((subject, img.tolist())))

umsgpack.pack(mean_image.tolist(), out)