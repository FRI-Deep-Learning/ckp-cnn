from PIL import Image, ImageFilter

def convert(imgName,resizeX,resizeY):
    rawImg = open(imgName,'rb').read()
    imgSize = (256,577)
    imgSize = (256,570)
    newImg = Image.frombytes('RGB',imgSize,rawImg,'raw')
    newImg = newImg.resize((resizeX,resizeY), Image.ANTIALIAS)
    newImg.save(img[:len(img)-4]+'.png')

img = 'test1.raw'
convert(img,96,96)
