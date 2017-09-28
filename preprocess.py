import numpy as np
import os
import cv2
from tqdm import tqdm

db_dir ="cohn-kanade-images"

def crop_to_face(img):
    cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    
    minisize = (img.shape[1],img.shape[0])
    miniframe = cv2.resize(img, minisize)

    faces = cascade.detectMultiScale(miniframe)

    if len(faces) < 1:
        print("FAILED ON IMAGE")
        exit(1)
    
    x, y, w, h = [ v for v in faces[0] ]
    cv2.rectangle(img, (x,y), (x+w,y+h), (255,255,255))

    return img[y:y+h, x:x+w]


for subject in tqdm(os.listdir(db_dir)):
    for root, dirs, files in os.walk(os.path.join(db_dir, subject)):
        for file in files:
            if file.startswith("face"):
                os.remove(os.path.join(root, file))
                continue
            
            if file.endswith(".png") and not file.startswith("face_"):
                img = cv2.imread(os.path.join(root, file))
                cropped_img = crop_to_face(img)
                resized_img = cv2.resize(cropped_img, (64, 64))
                cv2.imwrite(os.path.join(root, "face_" + file), resized_img)