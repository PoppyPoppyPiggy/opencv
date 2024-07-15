import cv2
import numpy as np

min_confidence = 0.5

# Load Yolo
# readNet  함수는 YOLOv3의 가중치 파일 ('yolov3.weights')과 
# 구성파일 ('yolov3.cfg')을 로드하여 신경망 객체 생성
# coco.names 파일을 읽어 객체 탐지에 사용할 클래스 이름들을 classes 리스트에 저장

net = cv2.dnn.readNet("C:/Users/rlaal/OPENCV/yolo/yolov3.weights", "C:/Users/rlaal/OPENCV/yolo/yolov3.cfg")
classes = []
with open("C:/Users/rlaal/OPENCV/yolo/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# net.getLayerNames 네트워크의 모든 레이어 이름을 가져옴
# net.getUnconnectedOutLayers() 출력 레이어의 인덱스 가져옴
# 출력 레이어의 이름을 output_layers 리스트에 저장
# 각 클래스에 랜덤 색상 저장 -> 객체 탐지 결과 시각화
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Loading image
# 이미지 로드및 이미지 비율을 가로 0.4 세로 0.4로 조정
img = cv2.imread("image/yolo_01.jpg")
img = cv2.resize(img, None, fx=0.4, fy=0.4)
height, width, channels = img.shape
cv2.imshow("original img", img)

# Detecting Object
# 이미지를 전처리하여 블롭(blob)을 생성
# 0.00392: 이미지 픽셀 값을 스케일링하기 위한 인자. 
# 일반적으로 1/255 (0.00392)로 설정하여 픽셀 값을 [0, 1] 범위로 변환합니다.
# (416, 416): YOLOv3 모델의 입력 크기. 이미지가 이 크기로 리사이즈됩니다.
# (0, 0, 0): 이미지 평균 값. 일반적으로 (0, 0, 0)으로 설정하여 이미지의 색상 공간을 변환하지 않습니다.
# True: BGR에서 RGB로 색상 공간 변환.
# crop=False: 크롭하지 않음. 이미지를 지정된 크기로 직접 리사이즈합니다.
# yolo는 데이터 타입 크기 320x320, 416x416, 609x609
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

net.setInput(blob)
outs = net.forward(output_layers)

# Showing information on the screen
class_ids = []
confidences = []
boxes = []

for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > min_confidence:
            # Object detected
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # Rectangle coordinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# 박스가 같은 얼굴을 계속 탐지하지 않도록 NMS 함수를 적용 시킴
indexes = cv2.dnn.NMSBoxes(boxes, confidences, min_confidence, 0.4)

font = cv2.FONT_HERSHEY_PLAIN
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        print(i,label)
        color = colors[class_ids[i]]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label, (x, y - 10), font, 1, color, 2)

cv2.imshow("yolo img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
