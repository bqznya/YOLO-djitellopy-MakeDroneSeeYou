# movement.py

from djitellopy import Tello
import configparser

def read_coordinates_from_cfg(file_path):
    config = configparser.ConfigParser()
    config.read(file_path)
    x = int(config.get('DEFAULT', 'x'))
    y = int(config.get('DEFAULT', 'y'))
    z = int(config.get('DEFAULT', 'z'))
    return x, y, z

def calculate_relative_coordinates(takeoff_coords, target_coords):
    return (
        target_coords[0] - takeoff_coords[0],
        target_coords[1] - takeoff_coords[1],
        target_coords[2] - takeoff_coords[2],
    )

def move_drone(tello, relative_target_coords, speed):
    tello.go_xyz_speed(*relative_target_coords, speed)

if __name__ == "__main__":
    takeoff_x, takeoff_y, takeoff_z = read_coordinates_from_cfg('takeoff.cfg')
    target_x, target_y, target_z = read_coordinates_from_cfg('target.cfg')

    relative_target_coords = calculate_relative_coordinates(
        (takeoff_x, takeoff_y, takeoff_z),
        (target_x, target_y, target_z)
    )

    tello = Tello()
    tello.connect()
    tello.takeoff()
    move_drone(tello, relative_target_coords, 20)
    tello.land()
    tello.end()
