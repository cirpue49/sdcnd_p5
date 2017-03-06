##Vehicle Detection Project
###Udacity Self-Driving Car Nanodegree Term 1 project 5
---


[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/hog.png
[image3]: ./examples/sliding.png
[image4]: ./examples/pipeline.png
[image5]: ./examples/img50.jpg
[image6]: ./examples/example_output.jpg
[video1]: ./project_video.mp4


###Histogram of Oriented Gradients (HOG)

####1. Explain how I extracted HOG features from the training images.

The code for this step is contained in the first code cell of the IPython notebook (or in lines # through # of the file called `some_file.py`).  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

####2. Explain how I settled on my final choice of HOG parameters.

I tried various combinations of parameters from `orientation = 6, 9, 12`, `Pixcel_per_cell = (8,8), (10, 10), (12, 12)` and `cells_per_block = (2, 2), (6, 6), (10, 10)`. I chose the paramtor based on a classifier peformance.

####3. Describe how I trained a classifier using my selected HOG features (and color features if you used them).

I trained a linear SVM using HOG features, Spatial features and histogram features. For HOG features, I chose YCrCb color space and extract HOG features from all color channel.

###Sliding Window Search

####1. Describe how I implemented a sliding window search.  How did I decide what scales to search and how much to overlap windows?

I combined 4 different size of windows. The point is to search windows under horizontal line since cars do not exist in the above part. In addition, I  serched relatively small windows around the horizontal line. 

```
big_windows = slide_window(image, x_start_stop=[None, None], y_start_stop=[400, 700], 
                    xy_window=(165, 144), xy_overlap=(0.8, 0.7))
middle_windows = slide_window(image, x_start_stop=[None, None], y_start_stop=[400, 700], 
                xy_window=(143, 116), xy_overlap=(0.8,0.6))
small_windows = slide_window(image, x_start_stop=[None, None], y_start_stop=[400, 550], 
                xy_window=(96, 72), xy_overlap=(0.7,0.7))
smaller_windows = slide_window(image, x_start_stop=[None, None], y_start_stop=[400, 450], 
                xy_window=(77, 62), xy_overlap=(0.9,0.8))
windows = small_windows + middle_windows + big_windows + smaller_windows
```

![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to try to minimize false positives and reliably detect cars?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result. In addition, I used a heat map to remove false positive. In the heat map, I set threshold 3 and sucessfully removed false positives.

![alt text][image4]

####3 Result

[Project Video](https://www.youtube.com/watch?v=Ywa9UDmwuj0&feature=youtu.be)

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/Ywa9UDmwuj0/0.jpg)](https://www.youtube.com/watch?v=vySgXdDJlrs)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Deep learning could make object detection more robust and speedy. After term 1, I want to implement state-of-the-art object detection in deep learning.


