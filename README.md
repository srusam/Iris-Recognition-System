# Iris-Recognition-System
Problem Statement: <br>
Iris recognition is concerned with security and the maintenance of privacy in the context of data. It must prohibit illegal intervention. The main problem in iris recognition is the accurate detection and segmentation of the iris, and then matching the extracted iris code with a database of pre-existing iris codes. <br><br>
Objectives: <br>
The objective of iris recognition is to accurately and efficiently identify individuals based on the unique pattern of their iris. This technology can be used for a variety of applications such as access control, secure identification, and fraud prevention. The main goal of iris recognition is to achieve high accuracy, low false acceptance rates, and low false rejection rates. <br><br>
Process Design: <br>
1. We start by setting the path to iris images folder. <br>
2. Then we read all the iris images and store them in a dictionary. <br>
3. By the following function we start the camera to capture the user’s iris. <br>
cap = cv2.VideoCapture(0) <br>
4. A frame is captured by using the cap.read() method. <br>
ret, frame = cap.read() <br>
5. We convert this frame to grey scale. <br>
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) <br>
6. After converting to grey scale we detect iris in the frame by using CascadeClassifier. <br>
7. If the iris is detected then we extract iris region of interest (ROI). <br>
8. Then we preprocess the ROI and compare the iris ROI with know iris images from the dataset using the correlation coefficient. <br>
9. If a match is found then the name of respective person will be displayed. <br>
