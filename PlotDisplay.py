import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Считывание данных из JSON-файла
def read_data_from_json(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    return data

# Извлечение координат x, y, z и времени из данных
def extract_coordinates_and_time(data):
    x_values = []
    y_values = []
    z_values = []
    time_values = []
    for entry in data:
        x_values.append(entry['x'])
        y_values.append(entry['y'])
        z_values.append(entry['z'])
        time_values.append(entry['time'])
    return x_values, y_values, z_values, time_values

# Построение трехмерного графика траектории движения дрона
def plot_trajectory_3d(x_values, y_values, z_values):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x_values, y_values, z_values, marker='o', linestyle='-')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Drone Trajectory in 3D Space')
    plt.show()

def main():
    filename = 'drone_data_2024-05-29_11-52-33_.json' # ???
    # Имя JSON-файла с данными
    # filename = 'drone_data_2024-05-30_14-35-50_.json' # удочка
    # filename = 'drone_data_2024-05-30_11-12-03_.json' # потолок
    # filename = 'drone_data_2024-05-29_14-12-54_.json' # куда
    # filename = 'drone_data_2024-05-29_13-17-22.json' # подъем по спиральке
    # filename = 'drone_data_2024-05-29_12-42-08.json' # не предел
    # filename = 'drone_data_2024-05-24_12-12-56.json' # default
    # filename = 'drone_data_2024-05-24_13-45-01_.json' # юла
    # Считывание данных из файла
    data = read_data_from_json(filename)

    # Извлечение координат x, y, z и времени
    x_values, y_values, z_values, time_values  = extract_coordinates_and_time(data)

    # Построение трехмерного графика траектории
    plot_trajectory_3d(x_values, y_values, z_values)

if __name__ == "__main__":
    main()
