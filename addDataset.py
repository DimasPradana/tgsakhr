'''
TODO:
    - warning ketika Id yg diinputkan sudah ada ✓
    - simpan gambar user per folder dengan nama folder == Id ✓
    - jika menggunakan kamera dengan resolusi kamera lebih besar,
    perlu pencahayaan yang baik ✗
    - jika inputan kosong, berikan error handle atau sediakan default
    values untuk gender dan crime_status pada variable ✗
    '''

from imutils.video import VideoStream
import numpy as np
import cv2 as cv
import imutils
import os
import time
import pymysql as psql


def add():

    vGender = None
    vCrime_status = None
    vId = input("Type your user Id: ")
    vName = input("Type your user Name: ")
    vGender = input("Input your Sex: ")
    vCrime_status = input("Input your Criminal State: ")
    print("[INFO] preparing connection database")
    db = psql.connect("localhost", "admin", "12345", "fr")
    cursor = db.cursor()
    dirname = vId
    dirCheck = os.path.exists("dataset/" + dirname)
    if dirCheck:
        exit()
        print("directory sudah ada")
    else:
        os.makedirs("dataset/" + dirname)
    sql = "INSERT INTO People (Id, Name, Gender, Crime_status)\
        VALUES (" + vId + ",'" + vName + "','" + str(vGender)\
        + "','" + str(vCrime_status) + "')"
    try:
        cursor.execute(sql)  # dimas
        db.commit()
    except Exception:
        db.rollback()
    print("[INFO] loading model...")
    net = cv.dnn.readNetFromCaffe("assets/deploy.prototxt.txt",
                                  "assets/res10_300x300_ssd_iter"
                                  "_140000.caffemodel")
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0, resolution=(640, 480)).start()
    time.sleep(2.0)
    sampleNum = 0
    while True:
        frame = vs.read()
        frame = imutils.resize(frame, width=800)
        (h, w) = frame.shape[:2]
        blob = cv.dnn.blobFromImage(cv.resize(frame, (300, 300)), 1.0,
                                    (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence < 0.5:
                continue
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            text = vName + " - {:.2f}%".format(confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            sampleNum = sampleNum + 1
            cv.imwrite("dataset/" + dirname + "/" + str(vId) + "." +
                       str(sampleNum) + ".jpg", frame)  # dimas
            cv.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv.putText(frame, text, (startX, y), cv.FONT_HERSHEY_SIMPLEX,
                       0.45, (0, 0, 255), 2)
        cv.imshow('Frame', frame)
        cv.waitKey(100)
        if sampleNum == 50:
            vs.stop()
            cv.destroyAllWindows()
            print("\n" + sql + "\n")
            print("Your input samples are: " + str(sampleNum))
            db.close
            break


if __name__ == '__main__':
    add()
