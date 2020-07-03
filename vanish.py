import numpy as np
import cv2

cap = cv2.VideoCapture(0)
fgbg = cv2.createBackgroundSubtractorMOG2()

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    #blur the frame to get rid of noise. the kernel should be ODD
    #frame = cv2.GaussianBlur(frame,(7,7),0)	

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #apply the background subtraction
    fgmask = fgbg.apply(frame)

    kernel = np.ones((3,3),np.uint8)
    kernel_lg = np.ones((7,7),np.uint8)
    #erosion followed by dilation is called an opening
    #mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
	
	#erode the mask to get rid of noise
    fgmask = cv2.erode(fgmask,kernel,iterations = 1)

    #dialate it back to regain some lost area
    fgmask = cv2.dilate(fgmask,kernel_lg,iterations = 1) 

    # Display the resulting frame
    cv2.imshow('frame',gray)
    cv2.imshow('vanish',fgmask)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()