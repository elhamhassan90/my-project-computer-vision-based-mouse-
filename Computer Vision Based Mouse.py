import cv2 as cv
import numpy as np 
import pyautogui
from matplotlib import pyplot as plt

cam = cv.VideoCapture(0)

lower_yellow = np.array([20,100,100])    ##Yellow_Move Cursor Pointer
upper_yellow = np.array([40,255,255])


lower_green = np.array([50,100,100])    ##Green _for Clicking(left)
upper_green = np.array([80,255,255])


lower_red = np.array([0,125,125])     ##red _for Clicking(right)
upper_red = np.array([10,255,255])


while(True):
    ret, frame = cam.read() ##in usual the cursor move in the opposite direction
    frame = cv.flip(frame, 1)  ## that to avoid that problem
    

    #smothen the image
    image_smooth = cv.GaussianBlur(frame,(7,7),0)

    #Definr Region Of Interest
    mask = np.zeros_like(frame)
    mask[50:350, 50:350] = [255, 255, 255]
    image_roi = cv.bitwise_and(image_smooth, mask)


    ## Drawing a rectangle on the image 
    cv.rectangle(frame, (50,50), (350,350), (0,0,255), 2)  
    cv.line(frame, (150,50), (150,350), (0,0,255), 1)
    cv.line(frame, (250,50), (250,350), (0,0,255), 1)
    cv.line(frame, (50,150), (350,150), (0,0,255), 1)
    cv.line(frame, (50,250), (350,250), (0,0,255), 1)
    


    ##Threshold the image for yellow color
    #img_hsv = cv.cvtColor(image_smooth, cv.COLOR_BGR2HSV)
    image_hsv = cv.cvtColor(image_roi, cv.COLOR_BGR2HSV)


    image_threshold = cv.inRange(image_hsv, lower_yellow, upper_yellow)

    #find contours
    contours, heirarchy = cv.findContours(image_threshold, \
                                                         cv.RETR_TREE, \
                                                         cv.CHAIN_APPROX_NONE)



    #find the index of the largest contour


    if(len(contours)!=0):
       areas = [cv.contourArea(c) for c in contours]
       max_index= np.argmax(areas)
       cnt = contours[max_index]
       #x_bound, y_bound, w_bound, h_bound = cv.boundingRect(cnt)
       #cv.rectangle(frame, (x_bound, y_bound), (x_bound + w_bound, y_bound + h_bound), (255,0,0), 2)

       ## pointer on video
       M = cv.moments(cnt)
       if (M['m00']!=0):
           cx = int(M['m10']/M['m00'])
           cy = int(M['m01']/M['m00'])
           cv.circle(frame, (cx,cy), 4, (0,255,0), -1) ##drawing a circle to represent the position of the marker

           #Cursor Motion
           if cx < 150:        ##if the pointer is in the left sector, the cursor should move towards the left.
               dist_x = -20
           elif cx > 250:      ##when the pointer is in the right sector, the cursor should move towards the right.
               dist_x = 20
           else:               ## the marker is in the central zone.
               dist_x = 0


           if cy < 150: 
               dist_y = -20
           elif cy > 250:      
               dist_y = 20
           else:               
               dist_y = 0
           pyautogui.moveRel(dist_x, dist_y, duration = 0.25)

           
      ## Check for (left)click
       image_threshold_green = cv.inRange(image_hsv, lower_green, upper_green)
       contours_green, heirarchy = cv.findContours(image_threshold_green, \
                                                                         cv.RETR_TREE, \
                                                                         cv.CHAIN_APPROX_NONE)

        
       if(len(contours_green) != 0):  ##check if the contours list is not empty, which means, a green colored contour is actually detected.
            pyautogui.click()
            cv.waitKey(1000)

    ## Check for click(right)
       image_threshold_red = cv.inRange(image_hsv, lower_red, upper_red)
       contours_red, heirarchy = cv.findContours(image_threshold_red, \
                                                                         cv.RETR_TREE, \
                                                                         cv.CHAIN_APPROX_NONE)
            

       if(len(contours_red) != 0):  ##check if the contours list is not empty, which means, a red colored contour is actually detected.
            pyautogui.click(button='right')
            cv.waitKey(1000)
                                                                         
           
    

    cv.imshow('Frame', frame)

    
    #key = cv.waitKey(10)
    key = cv.waitKey(100)
    if key == 27:
        break

#img_RGB= cv.cvtColor(frame, cv.COLOR_BGR2RGB)
#plt.imshow(img_RGB)
#plt.show()


cam.release()
cv.destroyAllWindows()
