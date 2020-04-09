import cv2 as cv
from imutils import face_utils
import dlib


#  cap = cv.VideoCapture(0)
#  while True:
#      _, image = cap.read()
#      gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
#
#      #  show the gray image
#      cv.imshow("Output", image)
#
#      #  key to give up the app
#      k = cv.waitKey(5) & 0xFF
#      if k == 27:
#          break
#
#  cv.destroyAllWindows()
#  cap.release()

def cek():

    p = "assets/shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(p)

    cap = cv.VideoCapture(0)

    while True:
        _, image = cap.read()
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        rects = detector(gray, 0)

        for (i, rect) in enumerate(rects):
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            for (x, y) in shape:
                cv.circle(image, (x, y), 2, (0, 255, 0), -1)

        cv.imshow("Frame", image)

        k = cv.waitKey(5) & 0xFF
        if k == 27:
            break

    cv.destroyAllWindows()
    cap.release()


if __name__ == '__main__':
    cek()
