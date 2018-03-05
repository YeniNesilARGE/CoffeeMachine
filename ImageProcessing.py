"""
This class is used as a layer between OpenCV and CoffeeMachine.
"""
class ImageProcessing:
    
    def __init__(self):
        pass

    # read a frame from video stream using by cv2.videoCapture() and return it.
    def catch_frame(self):
        pass

    # loads cascade classifiers in classifiers directory and put them in a dictionary.
    def load_cascades(self):
        pass

    # invoke classifiers for given frame
    def classifies(self, classifier, frame):
        pass

    # defines colors and returns their color spaces in RGB.
    def color_pack(self):
        pass

    # dilation, blurred, masking etc..
    def threshold(self, frame):
        pass

    # find all contours and return them
    def find_contours(self, frame):
        pass:

    # find convex hull for given contour
    def find_hull(self, contour):

    #drawings
    def draw_contour(self):
        pass
    def draw_hull(self):
        pass




