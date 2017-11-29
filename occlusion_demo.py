import cv2
import numpy as np
import random
import sys

DISPLAY_SIZE = 200

args = sys.argv[1:]

if len(args) < 1:
    print("Usage: python occlusion_demo.py <camera_number>")
    print("Start with 0 as the camera_number, and keep increasing it until the webcam is working.")
    exit(1)

random.seed()

def occlude_image_upper(image, rect):
	rx = random.randint(-3,4)
	ry = random.randint(-1,2)
	image[42+ry:42+rect.shape[0]+ry, 14+rx:14+rect.shape[1]+rx] = rect


def occlude_image_lower(image, rect):
	rx = random.randint(-3,4)
	ry = random.randint(-1,2)
	image[19+ry:19+rect.shape[0]+ry, 14+rx:14+rect.shape[1]+rx] = rect


def gen_gaussian_rect():
	h = random.randint(10,15)
	w = random.randint(36,43)
	rect = np.zeros((h,w))
	rmean = random.randint(20,131)
	cv2.randn(rect, rmean,30)

	return rect


def gen_saltandpepper_rect():
	h = random.randint(10,15)
	w = random.randint(36,43)
	rect = np.zeros((h,w))
	cv2.randu(rect,0,255)
	
	return rect


def occlude_image(image):
	im = np.asarray(image)
	im1 = im.copy()
	im2 = im.copy()
	im3 = im.copy()
	im4 = im.copy()

	# im1: upper, gaussian
	# im2: upper, sap
	# im3: lower, gaussian
	# im4: lower, sap

	occlude_image_upper(im1, gen_gaussian_rect())
	occlude_image_upper(im2, gen_saltandpepper_rect())
	occlude_image_lower(im3, gen_gaussian_rect())
	occlude_image_lower(im4, gen_saltandpepper_rect())

	return (im1, im2, im3, im4)

def capture_image():
    camera = cv2.VideoCapture(int(args[0]))
    while True:
        _, frame = camera.read()
        cv2.imshow("feed", frame)
        key = cv2.waitKey(1)
        if key == 13: # Enter key
            cv2.destroyWindow("feed")
            del(camera)
            return frame
        elif key == 27: # Escape key
            exit(0)

def display_image(image):
    cv2.imshow("image", image)
    if cv2.waitKey(0) == 27: # Escape key
        exit(0)

def crop_to_face(img):
    cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    
    minisize = (img.shape[1],img.shape[0])
    miniframe = cv2.resize(img, minisize)

    faces = cascade.detectMultiScale(miniframe)

    if len(faces) < 1:
        print("NO FACE!!")
        return None

    print("num faces:", len(faces))

    largest_face_idx = 0

    for idx, face in enumerate(faces):
        if face[2]*face[3] > faces[largest_face_idx][2]*faces[largest_face_idx][3]:
            largest_face_idx = idx
    
    x, y, w, h = [ v for v in faces[largest_face_idx] ]
    cv2.rectangle(img, (x,y), (x+w,y+h), (255,255,255))

    return img[y:y+h, x:x+w]


while True:
    img = capture_image()
    cv2.imwrite("test.png", img)

    if crop_to_face(img.copy()) is None:
        continue

    display_image(img)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    display_image(img)

    img = crop_to_face(img)
    display_image(img)

    img = cv2.resize(img, (64, 64))
    display_image(cv2.resize(img, (DISPLAY_SIZE, DISPLAY_SIZE)))

    (im1, im2, im3, im4) = occlude_image(img)

    def display_images(im1, im2, im3, im4):
        im1 = cv2.resize(im1, (DISPLAY_SIZE, DISPLAY_SIZE))
        im2 = cv2.resize(im2, (DISPLAY_SIZE, DISPLAY_SIZE))
        im3 = cv2.resize(im3, (DISPLAY_SIZE, DISPLAY_SIZE))
        im4 = cv2.resize(im4, (DISPLAY_SIZE, DISPLAY_SIZE))

        image = np.zeros((DISPLAY_SIZE*2, DISPLAY_SIZE*2), dtype="uint8")

        image[DISPLAY_SIZE:DISPLAY_SIZE*2, 0:DISPLAY_SIZE] = im1
        image[DISPLAY_SIZE:DISPLAY_SIZE*2, DISPLAY_SIZE:DISPLAY_SIZE*2] = im2
        image[0:DISPLAY_SIZE, 0:DISPLAY_SIZE] = im3
        image[0:DISPLAY_SIZE, DISPLAY_SIZE:DISPLAY_SIZE*2] = im4

        cv2.imshow("image", image)
        cv2.waitKey(0)

    display_images(im1, im2, im3, im4)

    cv2.destroyAllWindows()