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

    # open database connection
    print("[INFO] preparing connection database")
    db = psql.connect("localhost", "admin", "12345", "fr")
    cursor = db.cursor()

    # make a directory of Ids
    dirname = vId
    dirCheck = os.path.exists("dataset/" + dirname)
    # if(dirCheck == True):
    if dirCheck:
        exit()
        print("directory sudah ada")
    else:
        os.makedirs("dataset/" + dirname)

    # if not os.path.exists("dataset/" + dirname):
    # print("User ID has already used")
    # exit()
    # else:
    # os.makedirs("dataset/" + dirname)

    # insert into database
    sql = "INSERT INTO People (Id, Name, Gender, Crime_status)\
        VALUES (" + vId + ",'" + vName + "','" + str(vGender)\
        + "','" + str(vCrime_status) + "')"
    try:
        # execute the sql insert command
        cursor.execute(sql)  # dimas
        # commit your changes in the database
        db.commit()
    except Exception:
        # rollback in case there is any error
        db.rollback()

    # load model from disk
    print("[INFO] loading model...")
    net = cv.dnn.readNetFromCaffe("assets/deploy.prototxt.txt",
                                  "assets/res10_300x300_ssd_iter_14\
                                  0000.caffemodel")

    # initialize the video stream and warming up camera
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0, resolution=(640, 480)).start()
    time.sleep(2.0)

    # loop over the frames from the video stream
    sampleNum = 0
    while True:
        # grab the frame from the threaded video stream and resize it
        # to have a maximum width of 800 pixels
        frame = vs.read()
        frame = imutils.resize(frame, width=800)

        # grab the frame dimensions and convert it to blob
        (h, w) = frame.shape[:2]
        blob = cv.dnn.blobFromImage(cv.resize(frame, (300, 300)), 1.0,
                                    (300, 300), (104.0, 177.0, 123.0))
        # pass the blob through the network and obtain the detections
        # and predictions
        net.setInput(blob)
        detections = net.forward()

        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence associated with the prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence < 0.5:
                continue

            # compute the (x, y)-coordinate of the bounding box
            # for the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # draw the bounding box of the face along with the
            # associated probability
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
