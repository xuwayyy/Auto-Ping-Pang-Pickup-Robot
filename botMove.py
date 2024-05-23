import time

import numpy as np
from robomaster import robot
import cv2
import keyboard

rotate_time = 0


class DJRobot(robot.Robot):
    def __init__(self):
        super(DJRobot).__init__()
        # robomaster.config.LOCAL_IP_STR = "192.168.2.1"
        self.ep_robot = robot.Robot()

        self.ep_robot.initialize(conn_type="ap")

        self.ep_vision = self.ep_robot.vision

        self.ep_camera = self.ep_robot.camera

        self.ep_chassis = self.ep_robot.chassis

        self.ep_robot.set_robot_mode(mode=robot.CHASSIS_LEAD)
        # self.video_stream()
        # self.video_stream()

    def forward(self, dis):
        self.ep_chassis.move(x=dis, y=0, z=0, xy_speed=0.05).wait_for_completed()

    def forward_seconds(self, speed, duration):
        self.ep_chassis.drive_speed(x=speed, y=0, z=0, timeout=0.05)
        # time.sleep(duration)

    def stop_time(self):
        self.ep_chassis.drive_speed(x=0, y=0, z=0, timeout=10)

    def turn_right(self, angle):
        self.ep_chassis.drive_speed(x=0, y=0, z=-angle, z_speed=20).wait_for_completed()

    def turn_left_time(self, speed, duration):
        self.ep_chassis.drive_speed(x=0, y=0, z=-speed, timeout=0.05)
        # self.ep_chassis.move(x=0, y=0, z=-speed).wait_for_completed()
        # time.sleep(duration)

    def turn_right_time(self, speed, duration):
        self.ep_chassis.drive_speed(x=0, y=0, z=speed, timeout=0.05)
        # self.ep_chassis.move(z=speed).wait_for_completed()
        # time.sleep(duration)

    def turn_left(self, angle):
        self.ep_chassis.move(x=0, y=0, z=angle, z_speed=20).wait_for_completed()

    def video_stream(self):
        # self.ep_camera.start_stream()
        self.ep_camera.start_video_stream(display=False, resolution="720p")
        # print(self.ep_camera.video_stream_addr)

    def stop(self):
        # self.ep_chassis.stop()
        self.forward(0)

    def yolo_adjusting(self, center_x, area):
        global rotate_time
        if area < 146000:
            if center_x < 600:  # 620
                self.turn_left_time(4, 1)
                # while center_x < 640:

            elif center_x > 680: # 650
                # while center_x > 640:
                self.turn_right_time(4, 1)
            else:
                self.forward_seconds(0.05, 0.1)
                # self.stop()

        else:

            # self.stop_time()
            # for i in range(0, 10):  # 稳定一段时间
            #     # j = i
            #     time.sleep(1)
            time.sleep(15)
            self.stop()
            # if rotate_time < 4:
            #     self.turn_left(100)
            #     rotate_time += 1

    def hough_circle(self):
        self.ep_camera.start_video_stream(display=False, resolution="480p")
        global count
        global rotate_time
        while True:
            # print(count)
            if (keyboard.is_pressed('w') or keyboard.is_pressed('a') or keyboard.is_pressed('s') or keyboard.is_pressed(
                    'd')
                    or keyboard.is_pressed('e') or keyboard.is_pressed('q') or keyboard.is_pressed('space')):
                wasd(self.ep_chassis, self.ep_camera)
                continue
            image = self.ep_camera.read_cv2_image(strategy="newest")
            center_x = image.shape[1] // 2
            center_y = image.shape[0] // 2
            # cv2.imshow("frame", image)
            #
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            #
            blurred = cv2.medianBlur(gray, 7)

            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            # 定义黄色的HSV颜色范围
            lower_yellow = np.array([10, 100, 100])
            upper_yellow = np.array([30, 255, 255])

            # 使用颜色过滤来提取黄色区域
            mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
            cv2.imshow('mask', mask)
            cv2.imshow('image', image)
            # kernel = np.ones((5, 5), np.uint8)
            # opening = cv2.morphologyEx(blurred, cv2.MORPH_OPEN, kernel)

            circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, dp=1, minDist=28, param1=31, param2=13, minRadius=20,
                                       maxRadius=105)  # hsv param1 31, param2 15 20 105 # blur 140 42 20 105

            if circles is not None:
                circles = np.uint16(np.around(circles))
                count = 0

                for i in circles[0, :]:
                    # 绘制圆形边框
                    # cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)
                    # 绘制圆心
                    # cv2.circle(image, (i[0], i[1]), 2, (0, 0, 255), 3)
                    cv2.rectangle(image, (i[0] - i[2], i[1] - i[2]), (i[0] + i[2], i[1] + i[2]), (0, 0, 255), 2)

                    cv2.putText(image, f'({i[0]}, {i[1]})', (i[0] - 50, i[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (255, 255, 255), 2)
                    cv2.imshow('image', image)
                    r = i[2]
                    print("radius: ", r)
                    if r < 90:
                        adjusting(robot, (center_x, center_y), (i[0], i[1]))
                        # robot.forward(0.05)

            if circles is None:
                # robot.forward(0.1)
                count = count + 1
                print(count)
                threshold = 200
                if count > threshold:
                    if rotate_time < 4:
                        # time.sleep(30)
                        robot.turn_left(110)
                        count = 0
                        rotate_time = rotate_time + 1
                    else:
                        rotate_time = 0
                        robot.stop()
            if cv2.waitKey(1) & 0xFF == ord("p"):
                break

    def close(self):
        self.ep_camera.stop_video_stream()
        self.ep_robot.close()


def adjusting(robot: DJRobot, center: tuple, coordinate: tuple):
    center_x = center[0]

    x = coordinate[0]
    y = coordinate[1]
    interval = 20

    if x > center_x + interval:
        robot.turn_right_time(10, 1)
        # robot.turn_right(3)
    elif x < center_x - interval:
        robot.turn_right_time(10, 1)
        # robot.turn_left(3)
    else:
        # 方向调整完成
        # robot.forward(0.07)
        robot.forward_seconds(0.05, 0.1)
        robot.stop()
        # time.sleep(2)


def wasd(ep_chassis, ep_camera):
    x_val = 0.05
    y_val = 0.05
    z_val = 10

    image = ep_camera.read_cv2_image(strategy="newest")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #
    blurred = cv2.medianBlur(gray, 7)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 定义黄色的HSV颜色范围
    lower_yellow = np.array([10, 100, 100])
    upper_yellow = np.array([30, 255, 255])

    # 使用颜色过滤来提取黄色区域
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    cv2.imshow('mask', mask)
    cv2.imshow('image', image)
    # kernel = np.ones((5, 5), np.uint8)
    # opening = cv2.morphologyEx(blurred, cv2.MORPH_OPEN, kernel)

    circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, dp=1, minDist=28, param1=28, param2=12, minRadius=20,
                               maxRadius=135)  # hsv param1 31, param2 15 20 105 # blur 140 42 20 105

    if circles is not None:
        circles = np.uint16(np.around(circles))
        count = 0

        for i in circles[0, :]:
            # 绘制圆形边框
            cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # 绘制圆心
            cv2.circle(image, (i[0], i[1]), 2, (0, 0, 255), 3)
            cv2.putText(image, f'({i[0]}, {i[1]})', (i[0] - 50, i[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 2)
            cv2.imshow('image', image)
    cv2.imshow('image', image)

    time.sleep(0.02)
    if keyboard.is_pressed('w'):
        ep_chassis.drive_speed(x=x_val, y=0, z=0, timeout=0.05)
    elif keyboard.is_pressed('s'):
        ep_chassis.drive_speed(x=-x_val, y=0, z=0, timeout=0.05)
    elif keyboard.is_pressed('a'):
        ep_chassis.drive_speed(x=0, y=-y_val, z=0, timeout=0.05)
    elif keyboard.is_pressed('d'):
        ep_chassis.drive_speed(x=0, y=y_val, z=0, timeout=0.05)
    elif keyboard.is_pressed('e'):
        ep_chassis.drive_speed(x=0, y=0, z=z_val, timeout=0.05)
        # time.sleep(0.1)
    elif keyboard.is_pressed('q'):
        ep_chassis.drive_speed(x=0, y=0, z=-z_val, timeout=0.05)
    elif keyboard.is_pressed('space'):
        ep_chassis.drive_speed(x=0, y=0, z=0, timeout=0.05)
        time.sleep(10)


if __name__ == '__main__':
    robot = DJRobot()
    count = 0
    # rotate_time = 0
    # while True:
    #     wasd(robot.ep_chassis, robot.ep_camera)
    #
    #     if cv2.waitKey(1) & 0xFF == ord("p"):
    #         break
    # robot.hough_circle()
    # robot.close()
