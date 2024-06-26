from utlis import *
import cv2

w, h = 640, 480
pid = [0.4, 0.4, 0]
pError = [0, 0]
startCounter = 0  # for no Flight 1   - for flight 0

myDrone = initializeTello()

def telloGetFrame(myDrone, w=640, h=480):
    myFrame = myDrone.get_frame_read()
    myFrame = myFrame.frame
    img = cv2.resize(myFrame, (w, h))
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    return img

while True:
    ## Flight
    if startCounter == 0:
        myDrone.takeoff()
        myDrone.move_up(90)
        startCounter = 1

    ## Step 1
    img = telloGetFrame(myDrone, w, h)
    ## Step 2
    img, info = findFace(img)
    ## Step 3
    pError = trackFace(myDrone, info, w, h, pid, pError, inner_area=5000, outer_area=8000)
    # print(info[0][0])
    cv2.imshow('Image', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        myDrone.land()
        break
