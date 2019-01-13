# import the necessary packages

from picamera.array import PiRGBArray

from picamera import PiCamera

import time

import cv2

import os

import sys


def Cap():

	path = os.path.abspath(os.path.dirname(sys.argv[0]))

# initialize the camera and grab a reference to the raw camera capture

	camera = PiCamera()

	camera.resolution = (640, 480)

	camera.framerate = 32

	rawCapture = PiRGBArray(camera, size=(640, 480))



# allow the camera to warmup

	time.sleep(0.1)

# capture frames from the camera

	for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):        
	
		img_path = path +'/num.jpg'
	# grab the raw NumPy array representing the image, then initialize the timestamp

	# and occupied/unoccupied text
		image = frame.array

		cv2.rectangle(image,(int(200),int(100)),(int(284),int(184)),(255,255,255),4)

	# show the frame

		cv2.imshow("Capture img", image)
		key = cv2.waitKey(1) & 0xFF

        #select the box of img
		Cap = image[100:184,200:284]
		Cap = cv2.resize(Cap,(28,28))
		Cap = cv2.cvtColor(Cap,cv2.COLOR_BGR2GRAY)
		ret,Cap = cv2.threshold(Cap,127,255,cv2.THRESH_BINARY)
	
	
	# clear the stream in preparation for the next frame
		rawCapture.truncate(0)

		if key == ord("c"):
                    if os.path.isfile("num.jpg"):
                        os.remove("num.jpg")	                   
                        cv2.imwrite(img_path,Cap)
                    else:
                        cv2.imwrite(img_path,Cap)
			
	
	# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break

if __name__ == '__main__':
    Cap()

