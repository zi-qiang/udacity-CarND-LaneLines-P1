# **Finding Lane Lines on the Road** 

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./writeup-images/image1.png "Glitches"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 6 steps.

1. Convert the image to grayscale
2. Apply a Gaussian Noise kernel to the grayscale image
3. Apply Canny transform on the blur-image
4. Apply a quadrilateral mask to mask the region of interest
5. Apply Hough transform to identify lane lines
6. Draw the lines on the original image

In order to draw a single line on the left and right lanes, I modified the draw-lines() function by 

1. Separate the lines into left/right groups based on the line slope
2. In each line group, I take the 2 endpoints of each line. Then I use a *first degree least square polynomial fit* on those endpoints, to find the best-fit line.
3. Draw the line on the masked region.

There is some glitches on the lane lines while processing "solidYellowLeft.mp4", as there are sharp horizonal lines. as shown below
![alt text][image1]

To solve this issue. I apply a filter on the lines with __abs(slope) > 0.3__


### 2. Identify potential shortcomings with your current pipeline

The solution works fine with the first 2 videos, but fails miserably on the challenge video. Here are some potential shortcomings.

1. The mask region is tuned for the first 2 videos and does not apply well for the challenge video. Ideally, it should be smart enough to identify the optimal region of interest. Need a smarter adaptive algorithm.
2. The canny/hough transformation parameters shall be further tuned to fiter out the noisy lines(guardrails and tree shadows as appear on challenge.mp4).