'''
TODO:
    - jika ada lebih dari 1 wajah belum bisa ambil nama dari database ✓
    - counter untuk unknown ✓
    - addDataset ketika unknown lebih dari 10, tinggal rubah import ✓
    - print matchedidx atau counts untuk identifikasi wajah ✓ 133
    - print matches sebagai hasil compare menggunakan face_recognition ✓
    '''

from termcolor import colored
from imutils.video import VideoStream
import dlib_landmark as dl
import face_recognition
import imutils
import pickle
import time
import cv2
#  import pymysql as psql

global counter


def recognize():
    print(colored('[INFO] loading encodings...', 'red'))
    data = pickle.loads(open("assets/encodings.pickle", "rb").read())
    print(colored("[INFO] warming up camera...", 'yellow'))
    vs = VideoStream(src=0).start()
    time.sleep(2.0)
    global counter
    counter = 0
    while True:
        frame = vs.read()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = imutils.resize(frame, width=750)
        r = frame.shape[1] / float(rgb.shape[1])
        boxes = face_recognition.face_locations(rgb, model="cnn")
        encodings = face_recognition.face_encodings(rgb, boxes)
        names = []
        for encoding in encodings:
            matches = face_recognition.compare_faces(data["encodings"],
                                                     encoding)
            name = "Unknown"
            if True in matches:
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}
                for i in matchedIdxs:
                    name = data["names"][i]
                    counts[name] = counts.get(name, 0) + 1
                name = max(counts, key=counts.get)
            names.append(name)
        for ((top, right, bottom, left), name) in zip(boxes, names):
            top = int(top * r)
            right = int(right * r)
            bottom = int(bottom * r)
            left = int(left * r)
            #  text = "Unknown"
            text = "ini nomor {}".format(name)
            #  db = psql.connect("localhost", "admin", "12345", "fr")
            #  cursor = db.cursor()
            #  sql = "select * from People where Id = {}".format(name)
            #  try:
            #      cursor.execute(sql)
            #      results = cursor.fetchall()
            #      for result in results:
            #          vId = result[0]
            #          vName = result[1]
            #          vGender = result[2]
            #          vCrime_status = result[3]
            #          text = "{}, {}, {}, {}".format(
            #              vId, vName,
            #              vGender, vCrime_status)
            #          #  print(
            #          #      "\nmatchedIdxs: {}\ncounts: {}\
            #          #      \nenumerate: {}\nr: {}\n".
            #          #      format(matchedIdxs, counts, matches, r))
            #          #  print(name)
            #  except Exception:
            #      counter = counter + 1
            #      print("counter: {}, name: {}".format(counter, name))
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            y = top - 15 if top - 15 > 15 else top + 15
            cv2.putText(frame, text, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (0, 0, 255), 2)
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if counter == 10:
            cv2.destroyAllWindows()
            vs.stop()
            time.sleep(0.5)
            break
        elif key == ord("q"):
            cv2.destroyAllWindows()
            vs.stop()
            time.sleep(0.5)
            break


if __name__ == '__main__':
    recognize()
    if counter == 10:
        print("masuk ke 10")
        dl.cek()
    else:
        print(colored("[DONE] exiting...", 'green'))
