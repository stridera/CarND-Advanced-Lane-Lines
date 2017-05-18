import cv2
import numpy as np

class ThresholdProcessor():
    def __init__(self):
        ''' constructor '''

    def abs_sobel_thresh(self, image, orient='x', thresh=(0, 255)):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        if orient == 'x':
            abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
        if orient == 'y':
            abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))

        scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
        binary_output = np.zeros_like(scaled_sobel)
        binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
        return binary_output

    def mag_thresh(self, image, sobel_kernel=3, mag_thresh=(0, 255)):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        gradmag = np.sqrt(sobelx**2 + sobely**2)
        scale_factor = np.max(gradmag)/255
        gradmag = (gradmag/scale_factor).astype(np.uint8) 
        binary_output = np.zeros_like(gradmag)
        binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
        return binary_output

    def dir_threshold(self, image, sobel_kernel=3, thresh=(0, np.pi/2)):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
        binary_output =  np.zeros_like(absgraddir)
        binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
        return binary_output

    def hls_select(self, img, thresh=(0, 255)):
        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        s_channel = hls[:,:,2]
        binary_output = np.zeros_like(s_channel)
        binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
        return binary_output

    def processImage(self, image, combined_only=True):
        # Sorbel Kernel Size
        ksize = 17

        # Apply Sorbel Thresholding
        gradx = self.abs_sobel_thresh(image, orient='x', thresh=(10, 200))
        grady = self.abs_sobel_thresh(image, orient='y', thresh=(25, 200))
        # Get Magnitude and Gradient Directional Thresholding
        mag_binary = self.mag_thresh(image, sobel_kernel=ksize, mag_thresh=(30, 100))
        dir_binary = self.dir_threshold(image, sobel_kernel=ksize, thresh=(0.7, 1.3))
        hls_binary = self.hls_select(image, thresh=(90, 255))
        # Get a mask with pixels in both the x and y direction or with both a gradient/magnitude above the thresholds
        combined = np.zeros_like(dir_binary)
        combined[
            ((gradx == 1) & (grady == 1)) | 
            # ((mag_binary == 1) & (dir_binary == 1)) |
            (hls_binary == 1)
            ] = 1

        if combined_only:
            return combined
        else: 
            return gradx, grady, mag_binary, dir_binary, hls_binary, combined

def testImages(imgPath):
    from matplotlib import pyplot
    from CameraCalibrator import CameraCalibrator

    image = cv2.imread(imgPath)
    
    b,g,r = cv2.split(image)
    rgbImage = cv2.merge([r,g,b])

    calibrator = CameraCalibrator()
    calibrator.loadCalibrationValues()
    calibratedImage = calibrator.undistort(rgbImage)

    thresholdProcessor = ThresholdProcessor()
    gradx, grady, mag_binary, dir_binary, hsl_binary, combined = thresholdProcessor.processImage(rgbImage, combined_only = False)


    pyplot.subplot(331)
    pyplot.axis('off')
    pyplot.title("normal")
    pyplot.imshow(calibratedImage)

    pyplot.subplot(332)
    pyplot.axis('off')
    pyplot.title("gradx")
    pyplot.imshow(gradx, cmap='gray')

    pyplot.subplot(333)
    pyplot.axis('off')
    pyplot.title("grady")
    pyplot.imshow(grady, cmap='gray')

    pyplot.subplot(334)
    pyplot.axis('off')
    pyplot.title("mag_binary")
    pyplot.imshow(mag_binary, cmap='gray')

    pyplot.subplot(335)
    pyplot.axis('off')
    pyplot.title("dir_binary")
    pyplot.imshow(dir_binary, cmap='gray')

    pyplot.subplot(336)
    pyplot.axis('off')
    pyplot.title("hsl_binary")
    pyplot.imshow(hsl_binary, cmap='gray')

    pyplot.subplot(337)
    pyplot.axis('off')
    pyplot.title("combined")
    pyplot.imshow(combined, cmap='gray')

    pyplot.tight_layout()
    pyplot.show()

def testVideo(path):
    from moviepy.editor import VideoFileClip
    import os.path

    thresholdProcessor = ThresholdProcessor()

    filename, file_extension = os.path.splitext(path)
    challenge_output_path = "{}-processed{}".format(filename, file_extension)
    clip = VideoFileClip(path)
    processed_clip = clip.fl_image(thresholdProcessor.processImage)
    processed_clip.write_videofile(challenge_output_path, audio=False)


if __name__ == '__main__':
    # testImages('../test_images/straight_lines1.jpg')
    testImages('../test_images/bright_pavement.png')
    # testVideo("../short.mp4")

