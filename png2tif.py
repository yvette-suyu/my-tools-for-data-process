import PIL.Image
import sys
import numpy as np
import json, cv2, os

images_path="/media/hszc/data/syh/tesseract/data"

for x, _, z in os.walk(images_path):
    print (x)
    for name in z:
        if str(name).endswith(".png"):
            print (name)
            name_str = name[:-3]
            filename = images_path+ '/' + name
            print (filename)
            new_image = PIL.Image.open(filename)
            pre_savename = '/media/hszc/data/syh/tesseract/tif'
            new_image.save(os.path.join(pre_savename,os.path.basename(name_str+'.tif')))
