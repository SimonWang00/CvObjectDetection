import logging
import numpy as np
import cv2

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class ImageDetection():
	def __init__(self,confidence=0.5):
		# 置信度，默认0.5
		self.confidence = confidence
		self.prototxt = "./model/MobileNetSSD_deploy.prototxt.txt"
		self.model = "./model/MobileNetSSD_deploy.caffemodel"
		# 定义目标类别，飞机，自行车之类的
		self.CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat","bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
						"dog", "horse", "motorbike", "person", "pottedplant", "sheep","sofa", "train", "tvmonitor"]

	def loadImage(self,image_file):
		'''
		load image
		:param image_file:
		:return:
		'''
		image_np = cv2.imread(image_file)
		return image_np

	def detection(self,image_file):
		'''
		检测
		:param image_file:
		:return:
		'''
		# 方框颜色为随机生成
		COLORS = np.random.uniform(0, 255, size=(len(self.CLASSES), 3))
		image_np = self.loadImage(image_file)
		# 提取图片的高，框
		(h, w) = image_np.shape[:2]
		# 截取图片的300 x 300的图像区域，并转换成像素二进制储存在blob中
		blob = cv2.dnn.blobFromImage(cv2.resize(image_np, (300, 300)), 0.007843, (300, 300), 127.5)
		# 导入模型
		logging.info("--> 加载模型")
		net = cv2.dnn.readNetFromCaffe(self.prototxt, self.model)
		# 将存储的图片像素二进制数据输入神经网络
		net.setInput(blob)
		detections = net.forward()
		for i in np.arange(0, detections.shape[2]):
			# extract the confidence (i.e., probability) associated with the
			# prediction
			confidence = detections[0, 0, i, 2]    #算出置信度  检索顺序 第0个组，的第0个组，然后在第0个组的第三（2+1）个数
			# filter out weak detections by ensuring the `confidence` is
			# greater than the minimum confidence
			if confidence > self.confidence:
				# extract the index of the class label from the `detections`,
				# then compute the (x, y)-coordinates of the bounding box for
				# the object
				idx = int(detections[0, 0, i, 1])
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])  #0.34239042   0.30729017  0.6769601  0.82230425
				(startX, startY, endX, endY) = box.astype("int")  #进行数据变换 int型
				# display the prediction
				label = "{}: {:.2f}%".format(self.CLASSES[idx], confidence * 100)
				logging.info("[Detection] {}".format(label))
				# 在什么上面画一个，起始点是多少，终点是多少的，颜色是什么的，线条粗细的，矩形
				cv2.rectangle(image_np, (startX, startY), (endX, endY), COLORS[idx], 2)
				y = startY - 15 if startY - 15 > 15 else startY + 15
				cv2.putText(image_np, label, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
		#  在图片'image'上放置一个 “label”的文字内容，‘(startX, y)’为起始位置,字体为 FONT_HERSHEY_SIMPLEX，尺度为 0.5，颜色为‘idx’，磅数为2 的字符串
		# 输出图片名字是output的图片
		cv2.imshow("Output", image_np)
		logging.info("图片显示3秒钟后将自动关闭")
		cv2.waitKey(3000)


if __name__ == '__main__':
	test_image = './testSet/car.jpg'
	ImageDetection().detection(test_image)
