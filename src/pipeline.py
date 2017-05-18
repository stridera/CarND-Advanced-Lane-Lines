import cv2
import numpy as np
import os.path
from moviepy.editor import VideoFileClip


from CameraCalibrator import CameraCalibrator
from ThresholdProcessor import ThresholdProcessor
from PerspectiveTransformer import PerspectiveTransformer
from LaneLineFinder import LaneLineFinder

class LaneFindingPipeline():
	def __init__(self):
		self.calibrator = CameraCalibrator()
		self.calibrator.loadCalibrationValues()
		self.thresholdProcessor = ThresholdProcessor()
		self.perspectiveTransformer = PerspectiveTransformer()
		self.laneLineFinder = LaneLineFinder()
		self.frame = 0

	def overlay(self, image, ov):
		if (image.ndim == 2):
			image = np.asarray(np.dstack((image, image, image)), dtype=np.uint8)

		ov = ov[::5, ::5, ::5] # Shrink the image by 5x
		x_offset=image.shape[1] - ov.shape[1] - 25
		y_offset=25
		image[y_offset:y_offset+ov.shape[0], x_offset:x_offset+ov.shape[1]] = ov
		return image

	def overlayText(self, image, text, location, size=3, weight=8, color=(255,255,255)):
		font = cv2.FONT_HERSHEY_SIMPLEX
		cv2.putText(image, text, location, font, size, color, weight)
		return image

	def cleanup(self, image):
		return np.asarray(np.dstack((image, image, image)), dtype=np.uint8)*255

	def process(self, image, continuous = False):
		frames_per_step = 50
		self.frame = self.frame + 1
		bottom_right = (25, image.shape[0]-25)

		if self.frame < frames_per_step:
			return self.overlayText(image, "Default Video", bottom_right)

		''' Calibrate the camera and undistort the image '''
		undistortedImage = self.calibrator.undistort(image)

		if self.frame < frames_per_step * 2:
			return self.overlayText(undistortedImage, "Undistorted", bottom_right)

		''' Thresholding '''
		thresholdImage = self.thresholdProcessor.processImage(undistortedImage)

		if self.frame < frames_per_step * 3:
			im = self.cleanup(thresholdImage)
			im = self.overlay(im, undistortedImage)
			return self.overlayText(im, "Threshold Processing", bottom_right)
		
		''' Perspective Transform '''
		topdownView = self.perspectiveTransformer.apply_perspective_transformation(thresholdImage)
		
		if self.frame < frames_per_step * 4:
			im = self.cleanup(topdownView)
			# im = self.overlay(im, undistortedImage)
			return self.overlayText(im, "Topdown Perspective", bottom_right)
		
		''' Find lane lines '''
		visualization = 'final'
		text = "Final"
		overlay = True

		if self.frame < frames_per_step*5:
			visualization = 'colorize'
			text = 'Search for Lines'
			overlay = False
		elif self.frame < frames_per_step*6:
			visualization = 'colorize'
			text = 'Search for Lines'
			overlay = True
		elif self.frame < frames_per_step*7:
			visualization = 'lines'
			text = "Draw Lines"
			overlay = False
		elif self.frame < frames_per_step*8:
			visualization = 'lines'
			text = "Draw Lines"
			overlay = True

		lines = self.laneLineFinder.findLaneLines(topdownView, visualization)
		(left_curverad, right_curverad) = self.laneLineFinder.getCurves()
		curveText = "Right Curve: {0:.2f}m, Left Curve: {0:.2f}m".format(left_curverad, right_curverad)

		''' Restore View '''
		restoredView = self.perspectiveTransformer.reverse_perspective_transformation(lines)

		result = restoredView
		if overlay:
			''' Stack the restored image on top of the original '''
			result = cv2.addWeighted(undistortedImage, 1, np.uint8(restoredView), 1, 0)

		result = self.overlayText(result, curveText, (25, 25), 1, 3)

		return self.overlayText(result, text, bottom_right)

def processTestImages(pipeline):
	import glob
	from matplotlib import pyplot

	images = []

	for imgPath in glob.glob('../test_images/*'):
		images.append(imgPath)

	cols = len(images)
	for i, imgPath in enumerate(images):
		image = cv2.imread(imgPath)
		b,g,r = cv2.split(image)
		rgbImage = cv2.merge([r,g,b])

		processedImage = pipeline.process(rgbImage)


		# cv2.imwrite('../preprocessed_images/' + os.path.basename(imgPath), processedImage)

		pyplot.subplot(121)
		pyplot.axis('off')
		pyplot.title(imgPath)
		pyplot.imshow(rgbImage)
		pyplot.subplot(122)
		pyplot.title("{}, {}".format(cols*2, (i*2)))
		pyplot.imshow(processedImage)
	
		pyplot.tight_layout()
		pyplot.show()
		return

def processVideo(path, pipeline):
	filename, file_extension = os.path.splitext(path)
	challenge_output_path = "{}-processed{}".format(filename, file_extension)
	clip = VideoFileClip(path)
	processed_clip = clip.fl_image(pipeline.process)
	processed_clip.write_videofile(challenge_output_path, audio=False)


def main():
	''' Main Function '''
	pipeline = LaneFindingPipeline()

	# processTestImages(pipeline)
	processVideo("../project_video.mp4", pipeline)
	# processVideo("../short.mp4", pipeline)





if __name__ == '__main__':
	main()

