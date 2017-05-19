## Advanced Lane Lines

---

**Project Goals**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

**Notes**

* Code can be found here: [Advanced Lane Lines Github Source Directory](https://github.com/stridera/CarND-Advanced-Lane-Lines/tree/master/src)
* Each part of the pipeline (Calibration, Thresholding, Perspective transformation, and the Pipeline) live in their own class within the appropriate file.  Each class has a section that will test the class itself if it's run directly via the command line.

[//]: # (Image References)

[calibration1]: camera_fixed/calibration0.png "Undistorted 1"
[calibration2]: camera_fixed/calibration1.png "Undistorted 2"
[undistorted]: ./writeup_images/Undistorted.png "Distortion Correction Example"
[thresholding]: ./writeup_images/Thresholding.png "Thresholding Example"
[Perspective]:  ./writeup_images/PerspectiveTransformation.png "Perspective Transformation Example"
[final1]: ./output_images/straight_lines1.jpg "Final Example 1"
[final2]: ./output_images/straight_lines2.jpg "Final Example 2"
[video1]: ./project_video-processed.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.

You're reading it!  

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

Camera calibration happens inside the [CameraCalibrator.py](Camera Calibrator class) file.  This uses the OpenCV libraries to find the distortion, and assuming all images taken with the same camera have the same distortion values, undistort the images.

First you need to construct the class 

```python
	calibrator = CameraCalibrator()
```

Then you can calibrate it by providing a list of images.  In this case, we're using the images inside the camera_cal folder.  We call the calibrate function providing the chessboard size and give it a list of image paths.  We then save the values so they can be loaded later without having to go through the full calibration process each time.

```python
	for imgPath in glob.glob('../camera_cal/calibration*.jpg'):
		images.append(imgPath)
	
	if (calibrator.calibrate((9, 6), images)):
			calibrator.saveCalibrationValues()
```

Once saved, we can load them using the `loadCalibrationValues` function.  You can also print out the calibration settings using the `printCalibrationValues` function.  For our images we get the following values:

	mtx = [
	  [  8.26647419e+02,   0.00000000e+00,   4.00000007e+00],
	  [  0.00000000e+00,   2.78465905e+03,   2.50001140e+00],
	  [  0.00000000e+00,   0.00000000e+00,   1.00000000e+00]
	] 
	dist = [
	  [ -2.86633061e-02,   1.86159830e-04,   4.88453461e-04, 
	     8.28380180e-03,  -3.25418359e-07]
	]

Finally, we can use the undistort function to restore an image

```python
	for i, imgPath in enumerate(images):
		calibrator.undistort(image)
```
			
Behind the scenes, we calibrate by loading each image and convert it to grayscale.  Next we pass it on to `cv2.findChessboardCorners()` to get an array of discovered corners.  We then pass the found corners with an array formed using the original image points to `cv2.calibrateCamera()` to get the `mtx` and `dist` values shown above.  We are then able to pass those to `cv2.undistort()` to return an undistorted image.
			
![Calibration Example 1][calibration1] ![Calibration Example 2][calibration2]

[More calibration examples](https://github.com/stridera/CarND-Advanced-Lane-Lines/tree/master/camera_fixed)

In the end, we were unable to calibrate using images 1, 4, and 5.  This is because the images are too either too zoomed in or rotated so the findChessboardCorners function couldn't find the correct number of chessboard corners.  We still have the other 17 images, which is more than enough to calibrate the camera.

### Pipeline (single images)

Code: [pipeline.py](https://github.com/stridera/CarND-Advanced-Lane-Lines/blob/master/src/pipeline.py) in the `processTestImages()` function.

#### 1. Provide an example of a distortion-corrected image.
Here is an example of an image passed through the `CameraCalibration undistorted()` function described above.

![Undistorted Example Image][undistorted]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

Threshold Processing happens in the [ThresholdProcessor.py](https://github.com/stridera/CarND-Advanced-Lane-Lines/blob/master/src/ThresholdProcessor.py)

I have functions that do Sorbel Thresholding (shown as gradx and grady), Magnitude Thresholding (mag\_binary), Directional Thresholding (dir\_binary), and Saturation Color Thresholding (hsl\_binary).  I tried many different combinations to get the combined image before ending up with using a combination (AND) of the gradx and grady, OR hsl\_binary.  Processing is done on line 48 

![Thresholding Example Image][thresholding]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

Code at [PerspectiveTransformer.py](https://github.com/stridera/CarND-Advanced-Lane-Lines/blob/master/src/PerspectiveTransformer.py)

The first step in doing the perspective transformation was to figure out what points I wanted to capture and where I would stretch them out to.  I did this by opening up a test image inside an image manipulation program that shows cursor location and I noted the points for the lane.  I then ran it through the process a bunch of times tweaking the values until I had something that looked good.  I ended up using the following values:

    src = src or np.float32([[552, 460], [702, 460], [1100, 719], [205, 719]])
	dst = dst or np.float32([[153, 200], [981, 200], [941, 719], [313, 719]])

I could then get the transformation matrix (and corrisponding inverse matrix) by running the following:

```python
        self.matrix = cv2.getPerspectiveTransform(src, dst)
        self.inverse = cv2.getPerspectiveTransform(dst, src)
```

I could then get a topdown view, and restored view by calling `cv2.warpPerspective()` on the images, as shown on lines 13-17.

![Perspective Example Image][Perspective]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?
Lane line processing happens in [LaneLineFinder.py](https://github.com/stridera/CarND-Advanced-Lane-Lines/blob/master/src/LaneLineFinder.py).  Main processing function starts on line 70.

Using the instruction code, I was able to divide the screen into 9 windows and highlight the found lane lines by looking for the sections with the most bright pixel density.  (Histogram approach.)

On line 140, I take all the best fits, and run it through `np.polyfit()` to get the polynomials of the lines.  I take those numbers, and average them out with the last 15 frames to get a sliding window that can handle outliers.  


#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

Curvature and Center Finding happens in the [`updateCurvesAndCenter()`](https://github.com/stridera/CarND-Advanced-Lane-Lines/blob/master/src/LaneLineFinder.py#L183) function on line 183.  

We then do some guesswork, assuming its about 30 pixels per meter in the y axis, and 3.7 pixels per meter in the x axis and run it through these functions to get a curvature:

```python
self.left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
self.right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

```

Grabbing the center happens on line 214.  This part takes the center of the camera image and compares it against where we are in the lane.  We use a calculation we did earlier (line 151) to discover our center point.  basically we take the right line location and subtract the left lane.  This gives us the width of the lane.  Next we divide by two to get half the lane, and then shift it over to where the lane begins.  You can see this location by following the black line in the final images or video.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

Finally I overlay the lane information on the final image starting on line 163.  Here you can see the lane lines defined by the green overlay, with the center point defined by the black line.  

![Final Processing Example 1][final1]
![Final Processing Example 2][final2]

---

### Pipeline (video)


Code: [pipeline.py](https://github.com/stridera/CarND-Advanced-Lane-Lines/blob/master/src/pipeline.py) in the `processVideo()` function.

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

For a fun effect, I step through each stage in the first few frames of the video so you can see the processing that happens in each step.

Here's a [link to my video result](./project_video-processed.mp4).  You can also watch it via this [Youtube Link](https://youtu.be/Q3bwqF0o728).

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

**Problems and issues**

* I wish there was a better way to figure out the best threshold values and processes I should be using.  I still feel like I'm doing more guesstimate and use what looks right.
* Curvature values seem weird, but I'm not sure exactly how they should look.  Using code provided via the classroom work.  Probably more math here that I should understand and haven't been able to fully dive into.

**Future Enhancements**

* I really wanted to spend more time working on the line processing.  Both with more advanced validation and more efficient processing.