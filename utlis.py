from djitellopy import Tello
import cv2
import numpy as np

def initializeTello():
    myDrone = Tello()
    myDrone.connect()
    myDrone.for_back_velocity = 0
    myDrone.left_right_velocity = 0
    myDrone.up_down_velocity = 0
    myDrone.yaw_velocity = 0
    myDrone.speed = 0
    print(myDrone.get_battery())
    myDrone.streamoff()
    myDrone.streamon()
    return myDrone

def findFace(img):
    # Используем путь к стандартной директории OpenCV для каскада Хаара
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(imgGray, 1.1, 6)

    myFaceListC = []
    myFaceListArea = []

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cx = x + w // 2
        cy = y + h // 2
        area = w * h
        myFaceListArea.append(area)
        myFaceListC.append([cx, cy])

    if len(myFaceListArea) != 0:
        i = myFaceListArea.index(max(myFaceListArea))
        return img, [myFaceListC[i], myFaceListArea[i]]
    else:
        return img, [[0, 0], 0]

def trackFace(myDrone, info, w, h, pid, pError, inner_area, outer_area):
    x, y = info[0]
    area = info[1]

    ## PID для горизонтального движения
    error_x = x - w // 2
    speed_horizontal = pid[0] * error_x + pid[1] * (error_x - pError[0])
    speed_horizontal = int(np.clip(speed_horizontal, -100, 100))

    ## Определение движения вперед/назад
    if area > outer_area:
        myDrone.for_back_velocity = -20  # Дрон отдаляется
    elif area < inner_area and area != 0:
        myDrone.for_back_velocity = 20  # Дрон подлетает
    else:
        myDrone.for_back_velocity = 0  # Летает на месте

    ## Определение поворотов
    if speed_horizontal > 20:
        myDrone.yaw_velocity = 20  # Дрон поворачивается направо
    elif speed_horizontal < -20:
        myDrone.yaw_velocity = -20  # Дрон поворачивается налево
    else:
        myDrone.yaw_velocity = 0

    if myDrone.send_rc_control:
        myDrone.send_rc_control(myDrone.left_right_velocity,
                                myDrone.for_back_velocity,
                                myDrone.up_down_velocity,
                                myDrone.yaw_velocity)

    return [error_x, speed_horizontal]
