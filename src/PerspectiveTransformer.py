import cv2
import numpy as np

class PerspectiveTransformer():

    def __init__(self, src=None, dst=None): 
        ''' constructor '''
        src = src or np.float32([[552, 460], [702, 460], [1100, 719], [205, 719]])
        dst = dst or np.float32([[153, 200], [981, 200], [941, 719], [313, 719]])
        self.matrix = cv2.getPerspectiveTransform(src, dst)
        self.inverse = cv2.getPerspectiveTransform(dst, src)

    def apply_perspective_transformation(self, image):
        return cv2.warpPerspective(image, self.matrix, (image.shape[1], image.shape[0]))

    def reverse_perspective_transformation(self, image):
        return cv2.warpPerspective(image, self.inverse, (image.shape[1], image.shape[0])) 
 

if __name__ == '__main__':
    from matplotlib import pyplot
    from CameraCalibrator import CameraCalibrator

    testImagePath = '../test_images/straight_lines1.jpg'
    image = cv2.imread(testImagePath)
    
    b,g,r = cv2.split(image)
    rgbImage = cv2.merge([r,g,b])

    calibrator = CameraCalibrator()
    calibrator.loadCalibrationValues()
    calibratedImage = calibrator.undistort(rgbImage)

    perspectiveTransformer = PerspectiveTransformer()
    topdownView = perspectiveTransformer.apply_perspective_transformation(calibratedImage)
    restoredView = perspectiveTransformer.reverse_perspective_transformation(topdownView)

    pyplot.subplot(131)
    pyplot.axis('off')
    pyplot.title("Original")
    pyplot.imshow(calibratedImage)

    pyplot.subplot(132)
    pyplot.axis('off')
    pyplot.title("Top Down")
    pyplot.imshow(topdownView)

    pyplot.subplot(133)
    pyplot.axis('off')
    pyplot.title("Restored")
    pyplot.imshow(restoredView)

    pyplot.tight_layout()
    pyplot.show()
