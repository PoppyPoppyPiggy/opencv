import cv2
import face_recognition
import pickle # 정보 일렬로 serialize 하는거

dataset_paths = ['C:/Users/rlaal/OPENCV/opencv_test/opencv_data/dataset/son/',
                 'C:/Users/rlaal/OPENCV/opencv_test/opencv_data/dataset/tedy/']

names = ['Sons', 'Tedy']
number_images = 10
image_type = '.jpg'
encoding_file = 'C:/Users/rlaal/OPENCV/encodings.pickle'  # 절대 경로 사용
# cnn 느림 정확, hog 빠름 덜 정확
model_method = 'cnn'

knownEncodings = []
knownNames = []

for (i, dataset_path) in enumerate(dataset_paths):
    # 이름 가져옴
    name = names[i]
    
    for idx in range(number_images):
        file_name = dataset_path + str(idx + 1) + image_type
        
        # load convert it from bgr to dlib ordering
        image = cv2.imread(file_name)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        boxes = face_recognition.face_locations(rgb, model=model_method)
        encodings = face_recognition.face_encodings(rgb, boxes)
        
        for encoding in encodings:
            print(file_name, name, encoding)
            knownEncodings.append(encoding)
            knownNames.append(name)  # 이름 추가

# 인코딩 데이터 저장
data = {"encodings": knownEncodings, "names": knownNames}
with open(encoding_file, "wb") as f:
    f.write(pickle.dumps(data))
