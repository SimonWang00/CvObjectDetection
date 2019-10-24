#!/usr/bin/python3
# __*__ coding: utf-8 __*__

'''
@Author: simonKing
@License: (C) Copyright 2013-2019, Best Wonder Corporation Limited.
@Os：Windows 10 x64
@Contact: bw_wangxiaomeng@whty.com.cn
@Software: PY PyCharm 
@File: textDetection.py
@Time: 2019/10/23 17:22
@Desc: define your function
'''

from imutils.object_detection import non_max_suppression
import numpy as np
import logging
import cv2

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class TextDetection():
    def __init__(self,width=320,height=320,east="./model/frozen_east_text_detection.pb",min_confidence=0.5):
        self.width = width
        self.height = height
        self.east = east
        self.min_confidence = min_confidence
        pass

    def loadImage(self,image_file):
        '''
		load image
		:param image_file:
		:return:
		'''
        image_np = cv2.imread(image_file)
        return image_np

    def textDetection(self,image_file):
        image_np = self.loadImage(image_file)
        image_orig = image_np.copy()
        (H, W) = image_np.shape[:2]
        (newW, newH) = (self.width,self.height)
        rW = W / float(newW)
        rH = H / float(newH)
        # resize the image and grab the new image dimensions
        image_np = cv2.resize(image_np, (newW, newH))
        (H, W) = image_np.shape[:2]
        layerNames = ["feature_fusion/Conv_7/Sigmoid","feature_fusion/concat_3"]
        logging.info("--> loading EAST text detector...")
        net = cv2.dnn.readNet(self.east)
        # construct a blob from the image and then perform a forward pass of
        # the model to obtain the two output layer sets
        blob = cv2.dnn.blobFromImage(image_np, 1.0, (W, H),(123.68, 116.78, 103.94), swapRB=True, crop=False)
        net.setInput(blob)
        (scores, geometry) = net.forward(layerNames)
        (numRows, numCols) = scores.shape[2:4]
        rects = []
        confidences = []
        # loop over the number of rows
        for y in range(0, numRows):
            # extract the scores (probabilities), followed by the geometrical
            # data used to derive potential bounding box coordinates that
            # surround text
            scoresData = scores[0, 0, y]
            xData0 = geometry[0, 0, y]
            xData1 = geometry[0, 1, y]
            xData2 = geometry[0, 2, y]
            xData3 = geometry[0, 3, y]
            anglesData = geometry[0, 4, y]

            # loop over the number of columns
            for x in range(0, numCols):
                # if our score does not have sufficient probability, ignore it
                if scoresData[x] < self.min_confidence:
                    continue

                # compute the offset factor as our resulting feature maps will
                # be 4x smaller than the input image
                (offsetX, offsetY) = (x * 4.0, y * 4.0)

                # extract the rotation angle for the prediction and then
                # compute the sin and cosine
                angle = anglesData[x]
                cos = np.cos(angle)
                sin = np.sin(angle)

                # use the geometry volume to derive the width and height of
                # the bounding box
                h = xData0[x] + xData2[x]
                w = xData1[x] + xData3[x]

                # compute both the starting and ending (x, y)-coordinates for
                # the text prediction bounding box
                endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
                endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
                startX = int(endX - w)
                startY = int(endY - h)

                # add the bounding box coordinates and probability score to
                # our respective lists
                rects.append((startX, startY, endX, endY))
                confidences.append(scoresData[x])
        # apply non-maxima suppression to suppress weak, overlapping bounding
        # boxes
        boxes = non_max_suppression(np.array(rects), probs=confidences)
        # loop over the bounding boxes
        for (startX, startY, endX, endY) in boxes:
            # scale the bounding box coordinates based on the respective
            # ratios
            startX = int(startX * rW)
            startY = int(startY * rH)
            endX = int(endX * rW)
            endY = int(endY * rH)
            # draw the bounding box on the image
            cv2.rectangle(image_orig, (startX, startY), (endX, endY), (0, 255, 0), 2)
        logging.info("Text Detection Result:")
        cv2.imshow("Text Detection", image_orig)
        pname = './result/text1.jpg'
        cv2.imwrite(pname, image_orig)
        logging.info("Text Detection Result Saved. %s"%pname)
        logging.info("Window will close after 3 seconds")
        cv2.waitKey(3000)


if __name__ == '__main__':
    image_file = "./testSet/text1.jpg"
    TextDetection().textDetection(image_file)
