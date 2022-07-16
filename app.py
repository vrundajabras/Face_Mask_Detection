from flask import Flask, render_template, Response, request
import cv2
import datetime, time
import os, sys
import numpy as np
from threading import Thread
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import pymysql

global capture,rec_frame, grey, switch, neg, face, rec, out, mask 
capture=0
grey=0
neg=0
face=0
mask=0
switch=1
rec=0

connection = pymysql.connect( host="localhost", user="root", password="VDja06@*", database="students")
cursor=connection.cursor()

#make shots directory to save pics
try:
    os.mkdir('./shots')
except OSError as error:
    pass

#Load pretrained face detection model    
net = cv2.dnn.readNetFromCaffe('./saved_model/deploy.prototxt.txt', './saved_model/res10_300x300_ssd_iter_140000.caffemodel')
mask_net = load_model('./saved_model/mask_detector.model')

#instatiate flask app  
app = Flask(__name__, template_folder='./templates')


camera = cv2.VideoCapture(0)

def record(out):
    global rec_frame
    while(rec):
        time.sleep(0.05)
        out.write(rec_frame)

def detect_face(frame):
    global net
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0))   
    net.setInput(blob)
    detections = net.forward()
    confidence = detections[0, 0, 0, 2]

    if confidence < 0.5:            
            return frame           

    box = detections[0, 0, 0, 3:7] * np.array([w, h, w, h])
    (startX, startY, endX, endY) = box.astype("int")
    try:
        frame=frame[startY:endY, startX:endX]
        (h, w) = frame.shape[:2]
        r = 480 / float(h)
        dim = ( int(w * r), 480)
        frame=cv2.resize(frame,dim)
    except Exception as e:
        pass
    return frame

def detect_mask(frame, net, mask_net):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
		(104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    print(detections.shape)
    faces = []
    locs = []
    preds = []
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            faces.append(face)
            locs.append((startX, startY, endX, endY))
    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = mask_net.predict(faces, batch_size=32)
    
    return (locs, preds)

def gen_frames():  # generate frame by frame from camera
    global out, capture,rec_frame
    while True:
        success, frame = camera.read() 
        if success:
            if(face):                
                frame= detect_face(frame)
            if(mask):
                (locs, preds)= detect_mask(frame, net, mask_net)
                for (box, pred) in zip(locs, preds):
                    (startX, startY, endX, endY) = box
                    (mask_, withoutMask) = pred
                        
                    label = "Mask" if mask_ > withoutMask else "No Mask"
                    status_in_data= True if mask_ >withoutMask else False
                    sql=("INSERT INTO test(srno, res) VALUES (%s, %s)")
                    val=("7",status_in_data)
                    cursor.execute(sql,val)
                    cursor.execute("SELECT * FROM test")
                    result=cursor.fetchall()
                    connection.commit()
                    color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
                    label = "{}: {:.2f}%".format(label, max(mask_, withoutMask) * 100)
                    cv2.putText(frame, label, (startX, startY - 10),
			            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                    cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                cv2.imshow("Frame", frame)
                key = cv2.waitKey(1) & 0xFF
            if(capture):
                capture=0
                now = datetime.datetime.now()
                p = os.path.sep.join(['shots', "shot_{}.png".format(str(now).replace(":",''))])
                cv2.imwrite(p, frame)
                ##############################################################
                (locs, preds)= detect_mask(frame, net, mask_net)
                for (box, pred) in zip(locs, preds):
                    (startX, startY, endX, endY) = box
                    (mask_, withoutMask) = pred
                        
                    label = "Mask" if mask_ > withoutMask else "No Mask"
                    status_in_data= True if mask_ >withoutMask else False
                    sql=("INSERT INTO test(srno, res) VALUES (%s, %s)")
                    val=("8",status_in_data)
                    cursor.execute(sql,val)
                    cursor.execute("SELECT * FROM test")
                    result=cursor.fetchall()
                    connection.commit()
            
            try:
                ret, buffer = cv2.imencode('.jpg', cv2.flip(frame,1))
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                pass
                
        else:
            pass

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/requests',methods=['POST','GET'])
def tasks():
    global switch,camera
    if request.method == 'POST':
        if request.form.get('click') == 'Capture':
            global capture
            capture=1
        
        elif request.form.get('mask') == 'Detect Mask':
            global mask
            mask= not mask
                 
    elif request.method=='GET':
        return render_template('index.html')
    return render_template('index.html')

if __name__ == '__main__':
    app.run()