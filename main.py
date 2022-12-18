
# Python program for Detection of a
# specific color(blue here) using OpenCV with Python
import cv2
import numpy as np
import torch
import pandas as pd


# # Webcamera no 0 is used to capture the frames
# cap = cv2.VideoCapture('video.mp4')
#
#
# # This drives the program into an infinite loop.
# while(1):
#     # Captures the live stream frame-by-frame
#     _, frame = cap.read()
#     # Converts images from BGR to HSV
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#     lower_blue = np.array([110,50,50])
#     upper_blue = np.array([130,255,255])
#
#     # Here we are defining range of bluecolor in HSV
#     # This creates a mask of blue coloured
#     # objects found in the frame.
#     mask = cv2.inRange(hsv, lower_blue, upper_blue)
#     print(mask.shape)
#
#     # The bitwise and of the frame and mask is done so
#     # that only the blue coloured objects are highlighted
#     # and stored in res
#     res = cv2.bitwise_and(frame,frame, mask= mask)
#     cv2.imshow('frame',frame)
#     cv2.imshow('mask',mask)
#     cv2.imshow('res',res)
#
#     # This displays the frame, mask
#     # and res which we created in 3 separate windows.
#     k = cv2.waitKey(20) & 0xFF
#     if k == 27:
#         break
#
# # Destroys all of the HighGUI windows.
# cv2.destroyAllWindows()
#
# # release the captured frame
# cap.release()


import os
import cv2
import numpy as np
#read from tello video stream
cap = cv2.VideoCapture('video.mp4')
# model = torch.hub.load('ultralytics/yolov5', 'custom', path='last.pt')
# model = torch.hub.load( 'ultralytics/yolov5', 'custom', source = 'local', path = 'last.pt', force_reload = True)
model = torch.hub.load('/home/lab-maker/Desktop/yolov5', 'custom', path='last.pt', source = 'local')
model.conf = 0.7
# model = torch.hub.load('ultralytics/yolov5', "custom", 'last.pt')

# Check if camera opened successfully
if (cap.isOpened()== False):
  print("Error opening video stream or file")

# Read until video is completed
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:

    # Display the resulting frame
    results = model(frame) # batch of images
    # print(results.pandas().xyxy[0]["name"].to_list())
    df = results.pandas().xyxy[0]
    print(df)
    if "balloon" in df["name"].to_list():
      only = df[df["name"]== "balloon"]
      rel = only[["xmin", "ymin", "xmax", "ymax"]]
      rel = rel.values
      print(rel)
      print((rel[0][0] + rel[0][2]) // 2 ,  (rel[0][1] + rel[0][3]) // 2 )
      # for row in rel:
      #   hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
      #   lower_blue = np.array([103,132,0])
      #   upper_blue = np.array([161,255,255])
      #   mask = cv2.inRange(hsv, lower_blue, upper_blue)
      #   mask_row = mask[int(row[0]):int(row[2]), int(row[1]):int(row[3])]
      #   if np.sum(mask_row) > 1000000:
      #       # Window name in which image is displayed
      #       window_name = 'Image'
      #       # Start coordinate, here (5, 5)
      #       # represents the top left corner of rectangle
      #       start_point = (int(row[0]), int(row[1]))
      #       # Ending coordinate, here (220, 220)
      #       # represents the bottom right corner of rectangle
      #       end_point = (int(row[2]), int(row[3]))
      #       # Blue color in BGR
      #       color = (255, 0, 0)
      #
      #       # Line thickness of 2 px
      #       thickness = 2
      #
      #       # Using cv2.rectangle() method
      #       # Draw a rectangle with blue line borders of thickness of 2 px
      #       frame = cv2.rectangle(frame, start_point, end_point, color, thickness)
      #
      #       # Displaying the image
    cv2.imshow("img", frame)


    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break

  # Break the loop
  else:
    break

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()

#
# import cv2
# import numpy as np
#
# def nothing(x):
#     pass
#
# # Load image
# image = cv2.imread('20220918_194449.jpg')
# image = cv2.resize(image, (200,200))
#
# # Create a window
# cv2.namedWindow('image')
#
# # Create trackbars for color change
# # Hue is from 0-179 for Opencv
# cv2.createTrackbar('HMin', 'image', 0, 179, nothing)
# cv2.createTrackbar('SMin', 'image', 0, 255, nothing)
# cv2.createTrackbar('VMin', 'image', 0, 255, nothing)
# cv2.createTrackbar('HMax', 'image', 0, 179, nothing)
# cv2.createTrackbar('SMax', 'image', 0, 255, nothing)
# cv2.createTrackbar('VMax', 'image', 0, 255, nothing)
#
# # Set default value for Max HSV trackbars
# cv2.setTrackbarPos('HMax', 'image', 179)
# cv2.setTrackbarPos('SMax', 'image', 255)
# cv2.setTrackbarPos('VMax', 'image', 255)
#
# # Initialize HSV min/max values
# hMin = sMin = vMin = hMax = sMax = vMax = 0
# phMin = psMin = pvMin = phMax = psMax = pvMax = 0
#
# while(1):
#     # Get current positions of all trackbars
#     hMin = cv2.getTrackbarPos('HMin', 'image')
#     sMin = cv2.getTrackbarPos('SMin', 'image')
#     vMin = cv2.getTrackbarPos('VMin', 'image')
#     hMax = cv2.getTrackbarPos('HMax', 'image')
#     sMax = cv2.getTrackbarPos('SMax', 'image')
#     vMax = cv2.getTrackbarPos('VMax', 'image')
#
#     # Set minimum and maximum HSV values to display
#     lower = np.array([hMin, sMin, vMin])
#     upper = np.array([hMax, sMax, vMax])
#
#     # Convert to HSV format and color threshold
#     hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#     mask = cv2.inRange(hsv, lower, upper)
#     result = cv2.bitwise_and(image, image, mask=mask)
#
#     # Print if there is a change in HSV value
#     if((phMin != hMin) | (psMin != sMin) | (pvMin != vMin) | (phMax != hMax) | (psMax != sMax) | (pvMax != vMax) ):
#         print("(hMin = %d , sMin = %d, vMin = %d), (hMax = %d , sMax = %d, vMax = %d)" % (hMin , sMin , vMin, hMax, sMax , vMax))
#         phMin = hMin
#         psMin = sMin
#         pvMin = vMin
#         phMax = hMax
#         psMax = sMax
#         pvMax = vMax
#
#     # Display result image
#     cv2.imshow('image', result)
#     if cv2.waitKey(10) & 0xFF == ord('q'):
#         break
#
# cv2.destroyAllWindows()


