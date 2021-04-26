This folder contains data for Project 2 for the CSC420 class: http://www.cs.utoronto.ca/~fidler/teaching/2015/CSC420.html

The directory train and test contain the main data to work with. You are provided with the stereo images (left and right image), as well as annotations for the road. You are allowed to train your algorithms on 'train' but *not* on test. The results for which the project instructions ask for need to be done on 'test'. Each folder has image_left and image_right subfolders containing the images, a calib folder which contains the calibration (use devkit/loadCalibration.m to read it) and gt_image_left for ground-truth annotation for road.
More information about the format can be found in the devkit directory.

The directory train_angle contains images that you can use to train the car viewpoint classifiers. You are provided with images and labels. Each image has a corresponding mat file in the labels directory. The file contains a variable called 'annotation' which has several fields:

- class … class{i} is the class of the i-th object, e.g. Car, Van, DontCare, Pedestrian, Cyclist. You only care about the 'Car'
- bboxes …  bboxes(i, :) tell you [left,top,width,height] of the i-the object's bounding box. imshow(im); rectangle('position', bboxes(i,:)) will draw the box. 
- orient … where orient(i) is the viewpoint for the i-th object. It is in degrees, where 0 mean a Car facing you, 90 is a Car facing left, etc.
- truncated … truncated{i} tells you how much (from 0 to 1) the i-th object is truncated (outside the image). 
- occluded … occluded{i} tells you how much the i-th object is occluded. 0 for non-occluded, 1 for partially occluded, 2 and 3 highly occluded.
- box3D … where box3D{i} are the 3d bounding boxes around the objects in 3D, in case you need them
- boxView … where boxView{i} is the 3D object bounding box projected to the image plane. You can view it with the function devkit/plot_boxView.m
- camera contains internal parameters K and P in case you need them

IMPORTANT: Objects marked with 'DontCare' (class is DontCare) are objects that have not been annotated properly and thus you should skip them. Also ignore objects for which truncated > 0.3, and for which occluded > 2.

IMPORTANT2: Whenever you use code / data, you need to cite the paper that is in the corresponding README file.