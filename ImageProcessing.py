"""
This class is used as a layer between OpenCV and CoffeeMachine.
"""
import cv2
import numpy
class ImageProcessing:

    min_YCrCb = numpy.array([0,133,77],numpy.uint8)
    max_YCrCb = numpy.array([255,173,127],numpy.uint8)

    PATH_FACE = 'classifiers/haarcascade_frontalface_default.xml'
    PATH_SMILE = 'classifiers/smile.xml'
    
    def __init__(self):
        self.camera = cv2.VideoCapture(0)

        self.colors = color_pack

    # read a frame from video stream using by cv2.videoCapture() and return it.
    def catch_frame(self, flip = False):
        if not self.camera.isOpened():
            raise Exception('Camera is not opened')

        ret, frame = self.camera.read()
        if flip:
            frame = cv2.flip(frame, 1)
        return frame

    # loads cascade classifiers in classifiers directory and put them in a dictionary.
    def load_calissifiers(self):
        face_cascade = cv2.CascadeClassifier(PATH_FACE)
        smile_cascade = cv2.CascadeClassifier(PATH_SMILE)

        self.cascades = {}
        cascades['face'] = face_cascade
        cascades['smile'] = smile_cascade

    # invoke classify for given image
    def classify(self, classifier, img, scale_factor=1.1, min_neighbour=3):
        cascade = dict[classifier]

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        detected_objects = cascade.detectMultiScale(gray, scaleFactor, minNeighbour)

        return detected_objects        

    # defines colors and returns their color spaces in RGB.
    def color_pack(self):
        colors={}
        colors['red'] = (0, 0, 255)
        colors['blue'] = (255, 0, 0)
        colors['green'] = (0, 255, 0)
        colors['yellow'] = (0, 255, 255)
        colors['black'] = (0, 0, 0)
        colors['white'] = (255, 255, 255)
        colors['orange'] = (0, 165, 255)

        return colors

    # dilation, blurred, masking etc..
    def threshold(self, frame):
        YCrCb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)
        blurred = cv2.medianBlur(YCrCb, 5)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dilated = cv2.dilate(blurred, kernel)

        # using mask with pre-defined constants
        skin = cv2.inRange(dilated, self.min_YCrCb, self.max_YCrCb)

        return skin

    # find all contours and sort them in order to size of its area. If classifier is not optional
    # cascade get invokes and detecting objects will be extracted from contours. Function returns
    # contours and detecting objects.
    def find_contours(self, frame, classifier=None):
        _, contours, hierarchy = cv2.findContours(frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # sort contours in order to size of its area
        contours.sort(key=lambda contour: cv2.contourArea(contour))

        # if a classifier is given, invoke classifier and if an object detects in an any contour
        # extract the contour in contours list and put it an another list (the list would be 
        # usefull for the future)
        # this implemantation can be changed in future.

        detected_objects = []
        if classifier is not None:
            
            for i in range(len(contours)):
                cnt = contours[i]

                x, y, w, h = cv2.boundingRect(cnt)
                contour_area = frame[y:y+h, x:x+w]
                faces = classify('face')

                for face in faces:
                    draw_rectangle(frame, (x,y,w,h))
                    
                    f = list.pop(i)
                    detected_objects.append(f)

        return contours, detected_objects


    # find convex hull for given contour
    def find_hull(self, contour):
        hull = cv2.convexHull(contour)
        # cv2.drawContours(sourceImage, [hull], -1, (0, 0, 255), 2)

    #drawings
    def draw_contours(self, frame, contours, number_of_cnt=-1, color=None, thickness=2):
        if number_of_cnt is -1:
            cv2.drawContours(frame, contours, -1, self.colors['red'] if color is None else self.colors[color], thickness)
        else:
            for i in range(number_of_cnt):
                cv2.drawContours(frame, contours, i, self.colors['red'] if color is None else self.colors[color], thickness)
    def draw_hull(self, frame, hull, color=None, thickness=1):
        draw_contours(frame, [hull], -1, color, thickness)
    def draw_rectangle(self, frame, rect, color=None, thickness=2):
        for x,y,w,h in rect:
            cv2.rectangle(frame, (x, y), (x+w, y+h), self.colors['red'] if color is None else self.colors[color], thickness)




# sources: 
# https://github.com/seereality/opencvDemos/blob/master/skinDetect.py