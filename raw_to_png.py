from PIL import Image, ImageFilter
import os

path = 'C:/Users/IASA-FRI/Desktop/arfacetest/ARFaceDatabase/temp'
dest = 'C:/Users/IASA-FRI/Desktop/arfacetest/ARFaceDatabase/arface_png'

def rawToPng(imgDir,imgName,resizeX,resizeY,destDir):
    rawImg = open(imgDir + '/' + imgName,'rb').read()
    imgSize = (256,577)
    imgSize = (256,570)
    newImg = Image.frombytes('RGB',imgSize,rawImg,'raw')
    newImg = newImg.resize((resizeX,resizeY), Image.ANTIALIAS)
    newImg.save(dest + '/' + imgName[:len(img)-4]+'.png')

images = os.listdir(path)
for img in images:
    rawToPng(path,img,96,96,dest)
