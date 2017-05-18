import cv2
import numpy as np
import pickle

class CameraCalibrator:
	def __init__(self):
		self.calibrated = False

	def calibrate(self, chessboardShape, imagePaths):
		objPoints = [] #3D points in real world space
		imgPoints = [] #2D points in image plane

		cbx, cby = chessboardShape
		objp = np.zeros((cbx*cby, 3), np.float32)
		objp[:,:2] = np.mgrid[0:cby,0:cbx].T.reshape(-1,2)

		for imgPath in imagePaths:
			img = cv2.imread(imgPath)
			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			ret, corners = cv2.findChessboardCorners(gray, (cbx, cby), None)

			if ret:
				objPoints.append(objp)
				imgPoints.append(corners)
			else:
				print("Unable to find chessboard for image ", imgPath)

		if len(imgPoints) > 0:
			ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objPoints, imgPoints, chessboardShape, None, None)
			self.mtx = mtx
			self.dist = dist
			self.calibrated = True
			return True
		return False

	def undistort(self, image):
		if (self.calibrated):
			return cv2.undistort(image, self.mtx, self.dist, None, self.mtx)	
		else:
			raise CalibrationException("Calibrator not calibrated")

	def getCalibrationValues(self):
		if (self.calibrated):
			return self.mtx, self.dist
		else:
			raise CalibrationException("Calibrator not calibrated")

	def setCalibrationValues(self, mtx, dist):
		self.mtx = mtx
		self.dist = dist
		self.calibrated = True

	def saveCalibrationValues(self):
		with open('../data/camera_cal.dat', "wb") as f:
			pickle.dump(self.mtx, f)
			pickle.dump(self.dist, f)

	def loadCalibrationValues(self):
		with open('../data/camera_cal.dat', "rb") as f:
			self.mtx = pickle.load(f)
			self.dist = pickle.load(f)
			self.calibrated = True
			return True
		return False

	def printCalibrationValues(self):
		print(self.mtx, self.dist)

class CalibrationException(Exception):
	pass

# Test the class by running it directly.
if __name__ == '__main__':
	saveValues = False
	images = []

	import glob
	from matplotlib import pyplot

	calibrator = CameraCalibrator()

	for imgPath in glob.glob('../camera_cal/calibration*.jpg'):
		images.append(imgPath)

	if saveValues:
		if (calibrator.calibrate((9, 6), images)):
			calibrator.saveCalibrationValues()
	else:
		calibrator.loadCalibrationValues()

	if calibrator.calibrated:

		calibrator.printCalibrationValues()

		for i, imgPath in enumerate(images):
			image = cv2.imread(imgPath)
			pyplot.subplot(211)
			pyplot.axis('off')
			pyplot.title(imgPath)
			pyplot.imshow(image)
			pyplot.subplot(212)
			pyplot.axis('off')
			pyplot.title("corrected")
			pyplot.imshow(calibrator.undistort(image))
			pyplot.tight_layout()
			# pyplot.show()
			pyplot.savefig('../camera_fixed/calibration{}.png'.format(i), bbox_inches='tight')
	else:
		print("Calibrator not initialized.")

