'''
TODO:
    - jika ada lebih dari 1 wajah belum bisa ambil nama dari database ✓
    - counter untuk unknown ✓
    - addDataset ketika unknown lebih dari 10, tinggal rubah import ✓
    - print matchedidx atau counts untuk identifikasi wajah ✓ 133
    - print matches sebagai hasil compare menggunakan face_recognition ✓
    '''

# import the necessary packages
from termcolor import colored
from imutils.video import VideoStream
import dlib_landmark as dl
import face_recognition
import imutils
import pickle
import time
import cv2
import pymysql as psql


#  initiate global variable for counter
global counter


def recognize():
    # load the known faces and embeddings
    print(colored('[INFO] loading encodings...', 'red'))
    data = pickle.loads(open("assets/encodings.pickle", "rb").read())

    # initialize the video stream and pointer to output video file, then
    # allow the camera sensor to warm up
    print(colored("[INFO] warming up camera...", 'yellow'))
    vs = VideoStream(src=0).start()
    time.sleep(2.0)

    #  initiate global variable for counter inside function
    global counter
    counter = 0
    # loop over frames from the video file stream
    while True:
        # grab the frame from the threaded video stream
        frame = vs.read()

        # convert the input frame from BGR to RGB then resize it to have
        # a width of 750px (to speedup processing)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = imutils.resize(frame, width=750)
        r = frame.shape[1] / float(rgb.shape[1])

        # detect the (x, y)-coordinates of the bounding boxes
        # corresponding to each face in the input frame, then compute
        # the facial embeddings for each face
        boxes = face_recognition.face_locations(rgb, model="cnn")
        encodings = face_recognition.face_encodings(rgb, boxes)
        names = []

        # loop over the facial embeddings
        for encoding in encodings:
            # attempt to match each face in the input image to our known
            # encodings
            matches = face_recognition.compare_faces(data["encodings"],
                                                     encoding)
            name = "Unknown"

            # check to see if we have found a match
            if True in matches:
                # find the indexes of all matched faces then initialize a
                # dictionary to count the total number of times each face
                # was matched
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}

                # loop over the matched indexes and maintain a count for
                # each recognized face face
                for i in matchedIdxs:
                    name = data["names"][i]
                    counts[name] = counts.get(name, 0) + 1

                # determine the recognized face with the largest number
                # of votes (note: in the event of an unlikely tie Python
                # will select first entry in the dictionary)
                name = max(counts, key=counts.get)

            # update the list of names
            names.append(name)

        # loop over the recognized faces
        for ((top, right, bottom, left), name) in zip(boxes, names):
            # rescale the face coordinates
            top = int(top * r)
            right = int(right * r)
            bottom = int(bottom * r)
            left = int(left * r)

            #  connect to database
            text = "Unknown"
            db = psql.connect("localhost", "admin", "12345", "fr")
            cursor = db.cursor()
            sql = "select * from People where Id = {}".format(name)

            try:
                cursor.execute(sql)
                results = cursor.fetchall()
                for result in results:
                    vId = result[0]
                    vName = result[1]
                    vGender = result[2]
                    vCrime_status = result[3]
                    text = "{}, {}, {}, {}".format(
                        vId, vName,
                        vGender, vCrime_status)
                    #  print(
                    #      "\nmatchedIdxs: {}\ncounts: {}\
                    #      \nenumerate: {}\nr: {}\n".
                    #      format(matchedIdxs, counts, matches, r))
                    #  print(name)
            except Exception:
                #  print(sql)
                counter = counter + 1
                print("counter: {}, name: {}".format(counter, name))

            # draw the predicted face name on the image
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            y = top - 15 if top - 15 > 15 else top + 15
            #  cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
            cv2.putText(frame, text, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                        #  0.75, (0, 255, 0), 2)
                        0.75, (0, 0, 255), 2)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed,
        # and counter until 10 break from the loop
        if counter == 10:
            #  print("unknown sampai 10")
            # do a bit of cleanup
            cv2.destroyAllWindows()
            vs.stop()
            time.sleep(0.5)
            break
        elif key == ord("q"):
            # do a bit of cleanup
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
