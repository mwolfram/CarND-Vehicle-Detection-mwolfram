# CarND-Vehicle-Detection-mwolfram

## Writeup
---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Previous Assignment

The code from [Project 4 (Advanced Lane Lines)](https://github.com/mwolfram/CarND-Advanced-Lane-Lines-mwolfram) was reused in this project. The camera calibration step is described there. The lane line detection pipelines were reused for the videos, and the bounding boxes that resulted from vehicle detection were overlaid on these images.

### Configuration and RuntimeData

I used two singletons to
TODO describe both

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the ```single_img_features``` function (this in turn is called by by ```extract_features``` to extract features from multiple images). By activating ```hog_feat```, the function will extract hog features from the image and return the identified features as a feature vector.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

TODO two images here

![alt text][image1]

During the course I explored different color spaces and different hog parameters. I came up with "HLS" as color space, (16, 16) for spatial size and 16 histogram bins. I used all hog channels. Later in the project I realized that "YCrCb" yielded better results so that's used for the final video. In the "HLS" color space I had a significantly higher amount of false positives.

Here is an example using the following parameter set:

```python
self.COLOR_SPACE = "YCrCb"
self.SPATIAL_SIZE = (16, 16)
self.HIST_BINS = 16
self.ORIENTATIONS = 9
self.PIX_PER_CELL = 8
self.CELLS_PER_BLOCK = 2
self.HOG_CHANNEL = "ALL"
self.SPATIAL_FEAT = True
self.HIST_FEAT = True
self.HOG_FEAT = True
self.SCALE = 1.5
self.HIST_RANGE = (0, 1)
self.Y_START_STOP = [400, 670]
```


TODO sample hog features

![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

TODO very short trial and error

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

TODO continue here -------------------------------------

I trained a linear SVM using...

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
