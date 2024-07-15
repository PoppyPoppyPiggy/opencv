import cv2
import face_recognition
import pickle
import time
from tkinter import *
from PIL import Image
from PIL import ImageTk
from tkinter import filedialog

image_file = 'C:/Users/rlaal/OPENCV/opencv_test/opencv_data/image/marathon_01.jpg'
encoding_file = 'encodings.pickle'
unknown_name = 'unkownn'
model_method ='cnn'
title_name = 'face recognition'

def selectFile():
    print("select file")
    file_name =  filedialog.askopenfilename(initialdir = "./image",title = "Select file",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
    print('file name :',file_name)
    read_image = cv2.imread(file_name)
    image = cv2.cvtColor(read_image,cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    imgtk = ImageTk.PhotoImage(image = image)
    (height,width) = read_image.shape[:2]
    detectAndDisplay(read_image)



def detectAndDisplay(image):
    print("detect and display")
    start_time = time.time()
    rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    
    boxes = face_recognition.face_locations(rgb,model=model_method)
    encodings = face_recognition.face_encodings(rgb,boxes)

    names = []
    
    for encoding in encodings:
        mathces = face_recognition.compare_faces(data["encodings"], encoding)
        name = unknown_name
        
        if True in mathces:
            mathcedIdxs = [i for (i,b) in enumerate(mathces) if b]
            counts = {}
            
            for i in mathcedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name,0)+1
                
            name = max(counts,key=counts.get)
        names.append(name)
        
    for ((top,right,bottom,left),name) in zip (boxes,names):
        y = top-15 if top-15>15 else top+15
        color = (0,255,0)
        line = 2
        if (name==unknown_name):
            color = (0,0,255)
            line = 1
            name =''
        cv2.rectangle(image,(left,top),(right,bottom),color,line)
        y = top-15 if top-15>15 else top+15
        cv2.putText(image,name,(left,y),cv2.FONT_HERSHEY_SIMPLEX,0.75,color,line)
    
    end_time = time.time()
    process_time = end_time-start_time
    print("=== A frame took {:.3f} seconds".format(process_time))
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    imgtk = ImageTk.PhotoImage(image=image)
    detection.config(image = imgtk)
    detection.image = imgtk
#main
main = Tk()
main.title(title_name)
main.geometry()

data = pickle.loads(open(encoding_file,"rb").read())

read_image = cv2.imread(image_file)
image = cv2.cvtColor(read_image,cv2.COLOR_BGR2RGB)
image = Image.fromarray(image)
imgtk = ImageTk.PhotoImage(image=image)
(height, width) = read_image.shape[:2]

label = Label(main,text=title_name)
label.config(font=("Courier",18))
label.grid(row=0,column=0,columnspan=4)
Button(main,text="File Select", height=2,command=lambda:selectFile()).grid(row=1, column=0, columnspan=4, sticky=(W, E))
detection = Label(main,image=imgtk)
detection.grid(row=2,column=0,columnspan=4)
detectAndDisplay(read_image)

main.mainloop()