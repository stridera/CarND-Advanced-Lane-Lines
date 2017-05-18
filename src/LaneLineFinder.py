import cv2
import numpy as np

class Line():
	def __init__(self):
		# the confidence of the last line detected
		self.confidence = 0  
		# x values of the last n fits of the line
		self.recent_xfitted = [] 
		#average x values of the fitted line over the last n iterations
		self.bestx = None     
		#polynomial coefficients averaged over the last n iterations
		self.best_fit = None  
		#polynomial coefficients for the most recent fit
		self.current_fit = [np.array([False])]  
		#radius of curvature of the line in some units
		self.radius_of_curvature = None 
		#distance in meters of vehicle center from the line
		self.line_base_pos = None 
		#difference in fit coefficients between last and new fits
		self.diffs = np.array([0,0,0], dtype='float') 
		#x values for detected line pixels
		self.allx = None  
		#y values for detected line pixels
		self.ally = None

class LaneLineFinder:
	def __init__(self):
		''' Constructor '''
		self.frame = 0

		self.leftLine = Line()
		self.rightLine = Line()

		self.confidence = 0
		self.left_fit_list = []
		self.right_fit_list = []

		self.left_curverad = 0
		self.right_curverad = 0

		self.center_diff = 0

	def add_and_update_averages(self, left_fit, right_fit):
		window_size = 15

		if len(self.left_fit_list) < window_size:
			self.left_fit_list.append(left_fit)
		else:
			self.left_fit_list[self.frame % window_size] = left_fit

		if len(self.right_fit_list) < window_size:
			self.right_fit_list.append(right_fit)
		else:	
			self.right_fit_list[self.frame % window_size] = right_fit

		if self.frame == 1:
			return left_fit, right_fit

		return (np.average(self.left_fit_list, axis=0), np.average(self.right_fit_list, axis=0))

	def findLaneLines(self, image, visualize='final'):
		self.frame = self.frame + 1
		if (self.leftLine.confidence + self.rightLine.confidence > 100):
			return self.findLinesNearExisting(image, visualize)
		else:
			return self.findNewLines(image, visualize)


	def findNewLines(self, image, visualize='final'):
		# Taken from Udacity Instruction

		# Take a histogram of the bottom half of the image
		histogram = np.sum(image[image.shape[0]/2:,:], axis=0)
		# Create an output image to draw on and  visualize the result
		# out_img = np.dstack((image, image, image))*255
		out_img = np.zeros_like(np.dstack((image, image, image)))
		# Find the peak of the left and right halves of the histogram
		# These will be the starting point for the left and right lines
		midpoint = np.int(histogram.shape[0]/2)
		leftx_base = np.argmax(histogram[:midpoint])
		rightx_base = np.argmax(histogram[midpoint:]) + midpoint

		# Choose the number of sliding windows
		nwindows = 9
		# Set height of windows
		window_height = np.int(image.shape[0]/nwindows)
		# Identify the x and y positions of all nonzero pixels in the image
		nonzero = image.nonzero()
		nonzeroy = np.array(nonzero[0])
		nonzerox = np.array(nonzero[1])
		# Current positions to be updated for each window
		leftx_current = leftx_base
		rightx_current = rightx_base
		# Set the width of the windows +/- margin
		margin = 100
		# Set minimum number of pixels found to recenter window
		minpix = 50
		# Create empty lists to receive left and right lane pixel indices
		left_lane_inds = []
		right_lane_inds = []

		# Step through the windows one by one
		for window in range(nwindows):
			# Identify window boundaries in x and y (and right and left)
			win_y_low = image.shape[0] - (window+1)*window_height
			win_y_high = image.shape[0] - window*window_height
			win_xleft_low = leftx_current - margin
			win_xleft_high = leftx_current + margin
			win_xright_low = rightx_current - margin
			win_xright_high = rightx_current + margin

			# Draw the windows on the visualization image
			if (visualize == 'rectangles'):
				cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
				cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 

			# Identify the nonzero pixels in x and y within the window
			good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
			good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
			# Append these indices to the lists
			left_lane_inds.append(good_left_inds)
			right_lane_inds.append(good_right_inds)
			# If you found > minpix pixels, recenter next window on their mean position
			if len(good_left_inds) > minpix:
				leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
			if len(good_right_inds) > minpix:        
				rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

		# Concatenate the arrays of indices
		left_lane_inds = np.concatenate(left_lane_inds)
		right_lane_inds = np.concatenate(right_lane_inds)

		# Extract left and right line pixel positions
		leftx = nonzerox[left_lane_inds]
		lefty = nonzeroy[left_lane_inds] 
		rightx = nonzerox[right_lane_inds]
		righty = nonzeroy[right_lane_inds] 

		# Fit a second order polynomial to each
		left_fit = np.polyfit(lefty, leftx, 2)
		right_fit = np.polyfit(righty, rightx, 2)

		(left_fit_avg, right_fit_avg) = self.add_and_update_averages(left_fit, right_fit)

		# Generate x and y values for plotting
		ploty = np.linspace(0, image.shape[0]-1, image.shape[0] )

		left_fitx = left_fit_avg[0]*ploty**2 + left_fit_avg[1]*ploty + left_fit_avg[2]
		right_fitx = right_fit_avg[0]*ploty**2 + right_fit_avg[1]*ploty + right_fit_avg[2]
		center_fit = ((right_fitx - left_fitx) / 2) + left_fitx

		self.updateCurvesAndCenter(ploty, left_fit_avg, right_fit_avg, center_fit[-1])
		
		if (visualize == 'colorize'):
			out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
			out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

		if (visualize == 'lines'):
			cv2.polylines(out_img,  np.int32([np.stack((left_fitx, ploty), axis=-1)]), 0, (255,0,0), 30)
			cv2.polylines(out_img,  np.int32([np.stack((right_fitx, ploty), axis=-1)]), 0, (0,0,255), 30)

		if (visualize == 'final'):
			# Recast the x and y points into usable format for cv2.fillPoly()
			pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
			pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
			pts = np.hstack((pts_left, pts_right))

			# Draw the lane onto the warped blank image
			cv2.fillPoly(out_img, np.int_([pts]), (0,255, 0))

			cv2.polylines(out_img, np.int32([np.stack((center_fit, ploty), axis=-1)]), 0, (0,0,0), 30)


		return out_img

	def findLinesNearExisting(self, image):
		''' Todo '''

	def valid(self, left_fit, right_fit):
		''' Todo '''

	def updateCurvesAndCenter(self,ploty,left_fit,right_fit,center):
		# Define y-value where we want radius of curvature
		# I'll choose the maximum y-value, corresponding to the bottom of the image
		y_eval = np.max(ploty)
		left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
		right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])

		# Define conversions in x and y from pixels space to meters
		ym_per_pix = 30.0/720 # meters per pixel in y dimension
		xm_per_pix = 3.7/700 # meters per pixel in x dimension
		
		# Generate some fake data to represent lane-line pixels
		quadratic_coeff = 3e-4 # arbitrary quadratic coefficient
		leftx = np.array([200 + (y**2)*quadratic_coeff + np.random.randint(-50, high=51) for y in ploty])
		rightx = np.array([900 + (y**2)*quadratic_coeff + np.random.randint(-50, high=51) for y in ploty])
		leftx = leftx[::-1]  # Reverse to match top-to-bottom in y
		rightx = rightx[::-1]  # Reverse to match top-to-bottom in y
		# Fit a second order polynomial to pixel positions in each fake lane line
		left_fit = np.polyfit(ploty, leftx, 2)
		left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
		right_fit = np.polyfit(ploty, rightx, 2)
		right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

		# Fit new polynomials to x,y in world space
		left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
		right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
		# Calculate the new radii of curvature in meters
		self.left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
		self.right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

		# get Center
		camera_center = np.median(ploty) - 40
		self.center_diff = ((camera_center - center / 2) * xm_per_pix)


	def getCurvesAndCenter(self):
		return (self.left_curverad, self.right_curverad, self.center_diff)

if __name__ == '__main__':
	import glob
	from matplotlib import pyplot


	images = []
	for imgPath in glob.glob('../preprocessed_images/*'):
		images.append(imgPath)

	for i, imgPath in enumerate(images):
		image = cv2.imread(imgPath, 0)
		
		laneLineFinder = LaneLineFinder()
		processedImage = laneLineFinder.findLaneLines(image, 'final')


		pyplot.subplot(121)
		pyplot.axis('off')
		pyplot.title(imgPath)
		pyplot.imshow(image)
		pyplot.subplot(122)
		pyplot.axis('off')
		pyplot.imshow(processedImage)
	
		pyplot.tight_layout()
		pyplot.show()

		break

