# cctv_svm

### Calibration
* [python opencv calibration tutorial](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html)
* [opencv calibration docs](https://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html)
* [IPM](http://www.gisdeveloper.co.kr/?p=6832)


### How to use?
~~~
# for camera calibration
roslaunch cctv_svm calibration.launch

# for ipm
roslaunch cctv_svm ipm.launch

# for heading estimation
python3 Heading_estimation.py

# for empty space recognition
python3 Empty_space_recognition.py
~~~


### File Description

os : Ubuntu 16.04.2 LTS

GPU : GeForce GTX 970M (3GB)

Python : 3.5.2

Opencv : 3.4.0

|       File         |Description                                       |
|--------------------|--------------------------------------------------|
|Heading_estimation.py | Do heading estimation                                   |
|Empty_space_recognition.py | Do empty space recognition                                   |
|generator .py       |Synthetic Image Generator      |
|knn. py             |Training KNN classifier       |
|saved_model_3.pkl   |pre-trained KNN classifier                |
|detector.py         |Vehicle & Empty space Detection                           |

### topic
~~~
# calibration
publish : /distort_cam0, /distort_cam1

# ipm
subscribe : /distort_cam0, /distort_cam1
publish : /ipm0, /ipm1
~~~
