import robomaster
from robomaster import robot
tl_drone = robot.Drone()
tl_drone.initialize()
print("A")
# № 1: wifi = efc8fc
# № 2: wifi = 61e833
# 切换飞行器WiFi模式为组网模式，指定路由器SSID和密码
# tl_drone.config_sta(ssid="pasdemort", password="23613003")
tl_drone.config_sta(ssid="Galaxy", password="12345678")
# tl_drone.config_sta(ssid="bqznyatecnopova5", password="23613003")
print("A")
# tl_drone.config_sta(ssid="TP_Link_40", password="519936")
tl_drone.close()
print("A")