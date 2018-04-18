from __future__ import print_function
from __future__ import absolute_import
from torch.utils.data import Dataset
# import matplotlib.pyplot as plt
import sys, xml
import numpy as np
# import mxnet as mx
import json, cv2, os
import logging
import xml.etree.ElementTree as ET


class xmlDataset(Dataset):
    def __init__(self,
                 xml_path="/media/hszc/data/syh/tesseract/data",
                 images_path="/media/hszc/data/syh/tesseract/data",
                 ):
        self.images_path = images_path
        self.index = 0
        self.objs = []
        for x, _, z in os.walk(xml_path):
            for name in z:
                if str(name).endswith(".xml"):
                    oneimg = {}
                    oneimg['boxes'] = []
                    root = ET.parse(os.path.join(x, name)).getroot()
                    img_name = root.findall('filename')[0].text
                    oneimg['img_path'] = os.path.join(images_path, img_name)
                    for oobj in root.findall('object'):
                        name = oobj.find('name').text
                        bndbox = oobj.find('polygon')
                        point0 = bndbox.find('point0').text
                        point1 = bndbox.find('point1').text
                        point2 = bndbox.find('point2').text
                        point3 = bndbox.find('point3').text
                        point_str = point0 + "," + point1 + "," + point2 + "," + point3 + "," + name
                        oneimg['boxes'].append(point_str)
                    self.objs.append(oneimg)
        print("length,", len(self.objs))
    def __len__(self):
        return len(self.obj)

    def __getitem__(self, idx):
        return self.objs[idx]


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    reload(sys)
    sys.setdefaultencoding('utf-8')

    img_path = "/media/hszc/data/syh/tesseract/data"
    xml_path  = img_path
    output_path = "/media/hszc/data/syh/tesseract/418"
    xmldataset = xmlDataset(images_path=img_path,xml_path=xml_path)
    for oneimg in xmldataset:
        img_path = oneimg["img_path"]
        txt_path = os.path.join(output_path,os.path.basename(img_path)   + ".txt")
        logging.info("writting:{}".format(txt_path))
        with open(txt_path,"wt") as f:
            for polygon in oneimg["boxes"]:
                print(polygon,file=f)
                   # print(polygon)
