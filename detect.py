#############################################
# Universidad Tecnica Particular de Loja    #
#############################################
# Professor:                                #
# Rodrigo Barba        lrbarba@utpl.edu.ec  #
#############################################
# Students:                                 #
# Vanessa Narvaez     vlnarvaez@utpl.edu.ec #
# Sleyder Arteaga     sdarteaga@utpl.edu.ec #
#############################################

# import the necessary packages
from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2

# construct the argument parse and parse the arguments
cap = cv2.VideoCapture(0)

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

#parameters to record video
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
fps = 20
#record video
fourcc = cv2.VideoWriter_fourcc(*"XVID")	 
out = cv2.VideoWriter()
success = out.open("output.avi",fourcc,fps,size,True) 

# loop over the image paths
while(True):
    # read capture image
    ret, image = cap.read()
    # load the image and resize it to (1) reduce detection time
    # and (2) improve detection accuracy
    image = imutils.resize(image, width=min(500, image.shape[1]))
    orig = image.copy()

    # detect people in the image
    (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),
        padding=(8, 8), scale=1.05)

    # draw the original bounding boxes
    for (x, y, w, h) in rects:
        cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # apply non-maxima suppression to the bounding boxes using a
    # fairly large overlap threshold to try to maintain overlapping
    # boxes that are still people
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
    #assign tuplas to the lines	
    a = (250,0)
    b =	(250,400)
    c = (260,0)
    d =(260,400) 
    #draw the lines in the image
    cv2.line(image,a,b,(0,255,0),2)
    cv2.line(image,c,d,(0,255,0),2)
    # set counter
    cont=0	
    # draw the final bounding boxes
    for (xA, yA, xB, yB) in pick:
        cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)
	cv2.circle(image,(xA+35,yA+80), 10, (0,255,0), 0)
	#condition to count people crossing the line
    	if (xA+35,yA+80) > a and (xA+35,yA+80) <c:
	 cont = cont +1
    #present detections in the image
    font = cv2.FONT_HERSHEY_COMPLEX   	
    cv2.putText(image,"Personas:",(30,30), font, 1,(0,255,0),2)
    cv2.putText(image,str(cont),(200,30), font, 1,(0,255,0),2)  	  
    # show the output images
    if not ret:          
	break       
    out.write(image)
    cv2.imshow("ALGORITMO HOG", image)
    image = cv2.flip(image,1)		 	
    if(cv2.waitKey(1) & 0xFF == ord("q")):
        break
#release all     
cap.release()
out.release()
cv2.destroyAllWindows()
