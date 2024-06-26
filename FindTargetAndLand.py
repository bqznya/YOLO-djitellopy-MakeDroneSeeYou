from djitellopy import Tello
import cv2
import time
from ultralytics import YOLO
import numpy as np
import threading
import matplotlib.pyplot as plt
import configparser
import os

# Путь к модели YOLO
MODEL_PATH = 'best.pt'  # Путь к файлу обученной модели
# Загрузка модели YOLO
model = YOLO(MODEL_PATH)

# Инициализация дрона Tello
tello = Tello()
tello.connect()
tello.streamon()

# Глобальные переменные для обмена данными между потоками
frame = None
found_target = False
target_coords = None
current_height = None


CONFIG_FILE = 'target.cfg'

def read_coords():
    config = configparser.ConfigParser()
    if not os.path.exists(CONFIG_FILE):
        return [0, 0, 0]
    config.read(CONFIG_FILE)
    if 'DEFAULT' in config:
        x = config.getint('DEFAULT', 'x', fallback=0)
        y = config.getint('DEFAULT', 'y', fallback=0)
        z = config.getint('DEFAULT', 'z', fallback=0)
        return [x, y, z]
    return [0, 0, 0]

def write_coords(DEFAULT):
    config = configparser.ConfigParser()
    config['DEFAULT'] = {'x': str(int(DEFAULT[0])), 'y': str(int(DEFAULT[1])), 'z': str(int(DEFAULT[2]))}
    with open(CONFIG_FILE, 'w') as configfile:
        config.write(configfile)

def data_collection():
    global drone_positions, position_data
    last_time = time.time()
    plt.ion()  # Включение интерактивного режима
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    plt.title('Drone Trajectory')

    while True:
        current_time = time.time()
        elapsed_time = current_time - last_time
        last_time = current_time
        speed_x = tello.get_speed_x()
        speed_y = tello.get_speed_y()
        speed_z = tello.get_speed_z()
        drone_positions[0] += speed_x * elapsed_time / 100
        drone_positions[1] += speed_y * elapsed_time / 100
        drone_positions[2] += speed_z * elapsed_time / 100
        position_data["x"].append(drone_positions[0])
        position_data["y"].append(drone_positions[1])
        position_data["z"].append(drone_positions[2])

        # Запись координат в файл, перезаписывая его каждый раз
        with open('target.cfg', 'w') as file:
            file.write(f"x={drone_positions[0]:.2f}, y={drone_positions[1]:.2f}, z={drone_positions[2]:.2f}\n")

        ax.clear()
        ax.plot(position_data["x"], position_data["y"], position_data["z"], marker='o', linestyle='-')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Drone Trajectory')
        plt.draw()
        plt.pause(0.05)  # Небольшая пауза для обновления графика
        print(f"Drone Position: x={drone_positions[0]:.2f}, y={drone_positions[1]:.2f}, z={drone_positions[2]:.2f}")
        time.sleep(0.05)

        if tello.get_battery() < 10:  # Приземление при низком заряде батареи
            tello.land()
            break


# Функция для центрирования мишени в кадре
def center_target(tello, x1, y1, x2, y2, frame_width, frame_height):
    target_center_x = (x1 + x2) / 2
    target_center_y = (y1 + y2) / 2
    frame_center_x = frame_width / 2
    frame_center_y = frame_height / 2
    delta_x = target_center_x - frame_center_x
    delta_y = target_center_y - frame_center_y

    # Примерное значение для чувствительности коррекции
    sensitivity = 50

    # Корректировка по оси X
    if abs(delta_x) > sensitivity:
        if delta_x > 0:
            tello.send_rc_control(20, 0, 0, 0)  # Двигаться вправо
        else:
            tello.send_rc_control(-20, 0, 0, 0)  # Двигаться влево

    # Корректировка по оси Y
    if abs(delta_y) > sensitivity:
        if delta_y > 0:
            tello.send_rc_control(0, 0, 20, 0)  # Двигаться вниз
        else:
            tello.send_rc_control(0, 0, -20, 0)  # Двигаться вверх

# Функция для захвата и отображения видеопотока
def video_stream():
    global frame
    cap = tello.get_frame_read()
    while True:
        frame = cap.frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model.predict(source=frame, conf=0.25)
        if len(results) > 0 and results[0].boxes is not None:
            for det in results[0].boxes:
                x1, y1, x2, y2 = det.xyxy[0].tolist()
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        cv2.imshow('Drone Camera', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            tello.land()
            break

# Функция для распознавания объектов и управления дроном
def object_detection_and_control():
    global frame, found_target, target_coords, current_height
    rotation_b = False
    rotation_speed = 44  # Задайте скорость вращения по вашему усмотрению
    min_height = 20  # Минимальная высота для посадки

    tello.takeoff()
    print(tello.get_battery())
    tello.move_up(30)

    while True:
        if frame is None:
            continue

        # Обнаружение объектов в кадре с помощью модели YOLO
        results = model.predict(source=frame, conf=0.25)
        frame_height, frame_width, _ = frame.shape

        # Проверка на наличие обнаруженных объектов
        if len(results) > 0 and results[0].boxes is not None:
            found_target = False
            for det in results[0].boxes:
                # Извлечение координат ограничивающей рамки и уверенности
                x1, y1, x2, y2 = det.xyxy[0].tolist()
                conf = det.conf.item()
                cls = det.cls.item()

                # Предполагаем, что класс 0 - это посадочная площадка
                if int(cls) == 0 and conf > 0.75:  # Проверяем уверенность обнаружения
                    rotation_b = True
                    found_target = True
                    # Останавливаем дрон
                    tello.send_rc_control(0, 0, 0, 0)
                    # Рисование ограничивающей рамки
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                    # Центровка мишени в кадре
                    center_target(tello, x1, y1, x2, y2, frame_width, frame_height)
                    # Получение текущей высоты с датчика
                    # tello.move_down(20)
                    current_height = tello.get_height()
                    # Движение дрона вперед на рассчитанное расстояние
                    tello.move_forward(50)
                    if current_height > 29:
                        tello.move_down(20)

                    print("оп вниз вперед")
                    # Логика движения дрона
                    if current_height > 30:
                        # Проверяем позицию мишени
                        target_center_x = (x1 + x2) / 2
                        target_center_y = (y1 + y2) / 2
                        frame_center_x = frame_width / 2
                        frame_center_y = frame_height / 2

                        # Если мишень в центре кадра, летим вперед
                        if abs(target_center_x - frame_center_x) < 50 and abs(
                                target_center_y - frame_center_y) < 50:
                            tello.move_forward(50)
                            tello.move_down(30)
                            print("оп оп вниз вперед")

                        # Если мишень внизу кадра, летим вниз
                        elif abs(target_center_x - frame_center_x) < 50 and target_center_y > frame_center_y + 50:
                            if current_height > 29:
                                tello.move_down(30)
                            if current_height < 29:
                                tello.send_rc_control(0, 0, -20, 0)
                    else:
                        # Если достигнута минимальная высота, выполнить последний рывок вперед и приземлиться
                        print("Последний рывок")
                        #tello.move_forward(50)
                        tello.land()
                        break
                else:
                    tello.send_rc_control(0, 0, 0, 30)
            if not found_target:
                # Если не обнаружено объектов, дрон вращается на месте
                print("67")
                zxc = tello.get_height()
                if rotation_b:
                    if zxc < 10:
                        tello.send_rc_control(0, 0, 0, 30)
                        time.sleep(3)
                        tello.send_rc_control(0, 0, -10, 0)
                        time.sleep(2)
                    else:
                        tello.send_rc_control(0, 0, 0, 30)
                        time.sleep(3)
                        tello.send_rc_control(0, 0, -10, 0)
                        time.sleep(2)
                else:
                    if zxc < 10:
                        tello.send_rc_control(0, 0, 0, 40)
                        time.sleep(3)
                        tello.send_rc_control(0, 0, -10, 0)
                        time.sleep(2)
                    #                        tello.send_rc_control(0, 0, 0, rotation_speed)
                    else:
                        tello.send_rc_control(0, 0, 0, 40)
                        time.sleep(3)
                        tello.send_rc_control(0, 0, -10, 0)
                        time.sleep(2)
            #                       tello.send_rc_control(0, 0, -10, rotation_speed)


        else:
            # Если не обнаружено объектов, дрон вращается на месте
            print("71")
            tello.send_rc_control(0, 0, 0, rotation_speed)
        if frame is not None and found_target:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)


drone_positions = [0, 0, 0]

position_data = {"x": [drone_positions[0]], "y": [drone_positions[1]], "z": [drone_positions[2]]}


def visualize_trajectory():
    global drone_positions, position_data, stop_event
    last_time = time.time()
    plt.ion()
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    plt.title('Drone Trajectory')

    while not stop_event.is_set():
        current_time = time.time()
        elapsed_time = current_time - last_time
        last_time = current_time

        speed_x = tello.get_speed_x()
        speed_y = tello.get_speed_y()
        speed_z = tello.get_speed_z()

        drone_positions[0] += speed_x * elapsed_time / 100
        drone_positions[1] += speed_y * elapsed_time / 100
        drone_positions[2] += speed_z * elapsed_time / 100

        position_data["x"].append(drone_positions[0])
        position_data["y"].append(drone_positions[1])
        position_data["z"].append(drone_positions[2])

        write_coords(drone_positions)

        ax.clear()
        ax.plot(position_data["x"], position_data["y"], position_data["z"], marker='o', linestyle='-')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Drone Trajectory')
        plt.draw()
        plt.pause(0.05)
        write_coords(drone_positions)
        print(f"Drone Position: x={drone_positions[0]:.2f}, y={drone_positions[1]:.2f}, z={drone_positions[2]:.2f}")

        if tello.get_battery() < 10:
            tello.land()
            stop_event.set()
            break
        time.sleep(0.05)


# Создание и запуск потоков
video_thread = threading.Thread(target=video_stream)
data_thread = threading.Thread(target=data_collection)
control_thread = threading.Thread(target=object_detection_and_control)

data_thread.start()
video_thread.start()
control_thread.start()

data_thread.join()
video_thread.join()
control_thread.join()

# Отключение видеопотока и закрытие окон
tello.streamoff()
cv2.destroyAllWindows()
