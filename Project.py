import sys, os
sys.path.append(os.path.join(os.getcwd(),'python/'))

import numpy as np
import cv2
import darknet as dn
import pdb
import subprocess
from ctypes import *
import math
import random
import matplotlib.pyplot as plt
import glob

def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax

class Object:
    def __init__(self, label):
        self.label=label
        self.dct={}
    def addPosition(self,frameNo,posList):
        self.dct[frameNo]=posList

if __name__ == "__main__":
    imgs = glob.glob("data/*.jpg")
    images = sorted(imgs,key=lambda img: int(img.split("data/frame",1)[1].split(".jpg",1)[0]))
    print(images)
    net = dn.load_net(b"../cfg/yolov3.cfg", b"../yolov3.weights", 0)
    meta = dn.load_meta(b"../cfg/coco.data")
    #cap = cv2.VideoCapture('Dog.mp4')
    currentFrame = 0
    obj_list = []
    for frame in images:
        print(frame)
        img = bytes(frame,'utf-8')
        im = cv2.imread(frame)
        r = dn.detect(net, meta,img)
	#a = Pos
	#newObject = Object("person",list(3,4,5,6))
	#obj_list.append(newObject)
        print(r)
        newObjDistParam = 1   
        for i in r:
            x, y, w, h = i[2][0], i[2][1], i[2][2], i[2][3]
            xmin, ymin, xmax, ymax = convertBack(float(x), float(y), float(w), float(h))
            pt1 = (xmin, ymin)
            pt2 = (xmax, ymax)
            lbl = i[0].decode("utf-8")
            exist = False
            targetObject = None
            for obj in obj_list:
                if obj.label == lbl:
                    lst = list(obj.dct.keys())
                    print(lst)
                    
                    lastKey = lst[-1]
                    prevObjX = obj.dct[lastKey][0]
                    prevObjY = obj.dct[lastKey][1]
                    dist = math.sqrt(math.pow((y-prevObjX),2)+math.pow((x-prevObjY),2))
                    if dist > newObjDistParam:
                        newObject = Object(lbl)
                        newObject.addPosition(currentFrame,[x,y,w,h,i[1]])
                        obj_list.append(newObject)
                        targetObject = newObject
                        exist = True
                    else:
                        obj.addPosition(currentFrame,[x,y,w,h,i[1]])
                        exist = True
                        targetObject = obj
                    break
                        
            if not exist:
                newObject = Object(lbl)
                newObject.addPosition(currentFrame,[x,y,w,h,i[1]])
                obj_list.append(newObject)
                targetObject = newObject
            for obj in obj_list:
                print(obj.label,obj.dct)
            print("-----------------------------------------")
            boundary = 3
            distance = 0
            try:
                 currentFrameX = targetObject.dct[currentFrame][0]
                 currentFrameY = targetObject.dct[currentFrame][1]
                 previousFrameX = targetObject.dct[currentFrame-1][0]
                 previousFrameY = targetObject.dct[currentFrame-1][1]
                 distance = math.sqrt(math.pow((currentFrameY-previousFrameY),2)+math.pow((currentFrameX-previousFrameX),2))
            except:
                pass
            if distance <= boundary:
                 cv2.rectangle(im, pt1, pt2, (255, 0, 0), 2)
                 cv2.putText(im, i[0].decode() + " [" + str(round(i[1] * 100, 2)) + "]", (pt1[0], pt1[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, [0, 255, 0], 4)
            else:
                if currentFrame in targetObject.dct:
                    del targetObject.dct[currentFrame]
        #cv2.imshow("frame", im)
        #print (obj_frame)    
        name = './data/framedetect' + str(currentFrame) + '.jpg'
        print ('Creating...' + name)
        cv2.imwrite(name, im)
        currentFrame += 1
        k = cv2.waitKey(1)
        if k == 27:
            cv2.destroyAllWindows()
            exit()




  

    
