import numpy as np
import cv2

cap = cv2.VideoCapture(0)

if (cap.isOpened() == False):
    print("Unable to read camera feed")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("frame none")
        exit()

    # Our operations on the frame come here
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame', frame)
    # cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    # cv2.imshow('gray', cv2image)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
