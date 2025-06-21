# To show images in the window
import cv2

# Allows calculate fps
import time

# Importing the class which allows detect hands on the screen
from HandDetector_module import HandsDetector

# Getting access to camera
cap = cv2.VideoCapture(0)

################################################
wCam, hCam = 640, 480
################################################

# Setting the resolution of the video
cap.set(3, wCam)
cap.set(4, hCam)

# Variable for fps tracking
ptime = 0

while True:
    # Get frame from camera
    success, img = cap.read()
    
    # Stop the program if we do not have input image
    if not success:
        break

    # Counting fps
    ctime = time.time()
    fps = 1 / (ctime - ptime)
    ptime = ctime

    # Show fps on the window
    cv2.putText(img, f"FPS: {int(fps)}", (25, 75),
                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

    # Here we show result
    cv2.imshow("Result", img)
    
    # Quit the program by pressing q button
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()