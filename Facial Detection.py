import os
import imutils
import cv2 as cv 
from imutils import face_utils
from os import path
import numpy as np
import dlib
import math



''' This function takes in the video path and extracts frames from them
	After extracting frames, it performs landmark detection on each frame(frame by frame)
	Alongwith detecting features, it also classifies whether a person is speaking or not
	(i.e if his mouth is open or not)'''
def capture_frames(video_path):

	i = 1
	i_open = 1
	i_close= 1

	''' paths to store images of both the classes '''
	path_open = 'newopen/'
	path_close = 'newclose/'

	''' Capture Video '''
	cap = cv.VideoCapture(video_path)

	''' If read video fails''' 
	if not cap.isOpened():
		print("Error Reading File")
		quit(0)


	''' Capturing video frame by frame '''
	while i < cap.get(cv.CAP_PROP_FRAME_COUNT):
		ret, frame = cap.read()


		''' ret is True for correctly read frame '''
		if not ret:
			print("Error reading frame " + str(i))
			pass


		if int(cap.get(cv.CAP_PROP_POS_FRAMES)%20 == 0):

			''' Converting and Resizing the frame into grayscale '''
			gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

			gray_frame = imutils.resize(gray_frame, width=500)

			''' Detecting Landmarks '''
			point = landmark_pred(gray_frame)


			''' Classifying the landmarks based on the distance and
				saving the images in respective category folder'''
			if point is not None:
				if classify(point):
					cv.imwrite(os.path.join(path_open, video_path + 'frame' + str(i_open) + '.jpg'), gray_frame)
					print("Frame Open " + str(i_open))
					i_open+=1

				else:
					cv.imwrite(os.path.join(path_close, video_path + 'frame' + str(i_close) + '.jpg'), gray_frame)
					print("Frame Close " + str(i_close))
					i_close+=1
				i += 1
				print(point)
			else:
				pass
				
			
		else:
			pass

		if cv.waitKey(1) & 0xFF == ord('q'):
				break
				
	cap.release()
	cv.destroyAllWindows
	


''' This function takes an image and detects the landmarks of facial fetaures
	For our specific problem we are only detecting landmarks of the mouth. 
	After getting landmarks, we are calculating distance between "upper lip's lower part" 
	and "lower lip's upper part". That distance tells if the person is speaking or not'''
def landmark_pred(image):

	''' Initializing dlib's face detector (HOG based) and 
	then creating landmark face detector '''
	face_detector = dlib.get_frontal_face_detector()
	landmark_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


	''' Now detecting faces in the image '''
	detections = face_detector(image, 1)

	''' looping over face detections '''
	for (i, rect) in enumerate(detections):
		shape = landmark_predictor(image, rect)


		''' marking landmarks of mouth'''
		'''for i in range(60, 68):
			x = shape.part(i).x
			y = shape.part(i).y
			cv.circle(image, (x,y), 1, (0,0,255),-1)'''

		''' calculating distance between lips '''
		points = mouthCoordinates(shape)
		return points


''' This function calculates the distance between "upper lip's lower" part
	and "lower lip's upper" part'''
def mouthCoordinates(coordinates):

	x_y = np.empty([68, 2], dtype = int)
	for i in range(48, 68):
		x_y[i][0] = coordinates.part(i).x
		x_y[i][1] = coordinates.part(i).y


	m60, m61, m62, m63, m64, m65, m66, m67 = 0,0,0,0,0,0,0,0

	m60 = x_y[59]
	m61 = x_y[60]
	m62 = x_y[61]
	m63 = x_y[62]
	m64 = x_y[63]
	m65 = x_y[64]
	m66 = x_y[65]
	m67 = x_y[66]
	m68 = x_y[67]

	distance_m62_m68 = calculate_distance(m62, m68)
	distance_m63_m67 = calculate_distance(m63, m67)
	distance_m64_m66 = calculate_distance(m64, m66)

	return np.array([distance_m62_m68, distance_m63_m67, distance_m64_m66])



''' function to calculate distance ''' 
def calculate_distance(x, y):
	distance = np.linalg.norm(x-y)
	return distance


''' function to classify between mouth open and close'''
def classify(co_ords):

	''' threshold is zero because when a mouth is closed
		the distance between lips is zero'''
	threshold = np.array([0, 0, 0])

	if np.all(co_ords > threshold):
		return True

	else:
		return False


''' main function '''
while True:
	video_path = input("Enter video/path to extract Images\nPress q to exit\n")
	if not path.exists(video_path):
		if video_path == 'q':
			print("Quitting")
			break
		else:
			print("Enter a valid path/video name")

	else:
		capture_frames(video_path)
